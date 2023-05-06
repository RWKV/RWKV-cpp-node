//---------------------------
// Dependencies
//---------------------------

const os = require("os");
const tokenizer = require("rwkv-tokenizer-node");
const cpp_bind = require("./cpp_bind");
const ai_utils = require("./ai_utils");
const LRUCache = require('lru-cache');

//
// Token cache offset which we use
// this is used to intentionally to avoid caching the last few tokens of any given input
// which may be represented by a different token value, when merged with a larger input
//
const TOKEN_CACHE_OFFSET = 2;

//
// Minimum output token size for buffer eligibility, for output buffering
// This should strictly be greater than the TOKEN_CACHE_OFFSET by a few tokens
//
const MIN_OUTPUT_TOKEN_SIZE_FOR_BUFFERING = 5;

//---------------------------
// Implementation
//---------------------------

/**
 * RWKV js class wrapper
 * 
 * Allows the use of the RWKV CPP library from JS
 * This library is used to load and infer the RWKV model
 * 
 * It should not be used for training or model conversion
 */
class RWKV {

	//-------------
	// Class Setup
	//-------------

	/**
	 * Constructor, with the RWKV CPP model path
	 * 
	 * If initialized with a string, it is assumed to be the model path
	 * 
	 * If initialized with an object, you can use the following parameters
	 * 
	 * - path: the model path
	 * - threads: the number of threads to use 
	 *            (defaults to the number of vCPUs)
	 * - stateCacheSize: the hidden state cache size to use, 
	 *                   useful to speed up inference of a chat like model
	 *                   (defaults to 50)
	 * 
	 * @param {Object|string} config object, or the model path string
	 */
	constructor(config) {
		// Check if the config is a string, if so normalize it to a config obj
		if (typeof config === "string") {
			config = { path: config };
		}

		// Store the used config
		this._config = config;

		// Get the CPU thread count
		let threads = config.threads;
		if( threads == null || threads <= 0 ) {
			threads = os.cpus().length;
		}

		// Load the cpp context
		let ctx = cpp_bind.rwkv_init_from_file(config.path, threads);

		// Get the state and logits size
		this._state_size = cpp_bind.rwkv_get_state_buffer_element_count(ctx);
		this._logits_size = cpp_bind.rwkv_get_logits_buffer_element_count(ctx);

		// Store the context
		this._ctx = ctx;

		// Prepare the LRU cache with the configured cache size
		//
		// it is worth noting that the 7B model takes up about 2.64 MB for the state buffer, 
		// meaning you will need atleast 264 MB of RAM for a cachesize of 100

		// State cache, keeps the last N states in memory
		if( config.stateCacheSize === false || config.stateCacheSize <= 0 ) {
			// Disable the cache
			this._stateCache = null;
		} else {
			// Create the cache
			this._stateCache = new LRUCache({
				max: config.stateCacheSize || 50
			});
		}
	}

	/**
	 * Cleanup the RWKV context
	 */
	free() {
		// Destroy the context
		if( this._ctx ) {
			cpp_bind.rwkv_free(this._ctx);
			this._ctx = null;
		}
	}

	/**
	 * Internal function, throw if the context is not set
	 */
	_checkContext() {
		if( !this._ctx ) {
			throw new Error("RWKV context is not set, did you call free()?");
		}
	}

	//-------------
	// Internal ops
	//-------------

	/**
	 * @private mehthod (do not use API directly, it will not be maintained)
	 * 
	 * Given the existing state, and the input string, get the completed hidden state buffer.
	 * This operation DOES NOT use the internal cache.
	 * 
	 * @param {Float32Array} state - existing state to compute from
	 * @param {Array<number>} tokenArr - array of tokens to compute
	 * 
	 * @returns {Object} the hidden state buffer and logits buffer
	 */
	_getHiddenState_fromExistingState_andTokenArr(state, tokenArr) {
		// If state is null, we start from scratch
		if( state == null ) {
			state = new Float32Array(this._state_size);
		}

		// Edge case handling when tokenArr is empty
		if( tokenArr == null || tokenArr.length == 0 ) {
			throw new Error("RWKV token array is empty");
		}

		// Prepare the output state to use
		let outputState = new Float32Array(this._state_size);

		// Copy the state into the output state
		outputState.set(state);

		// Prepare the logit buffer (to be ignored sadly)
		let logits = new Float32Array(this._logits_size);

		// Compute the hidden state for each token
		for( const token of tokenArr ) {
			if( cpp_bind.rwkv_eval(
				this._ctx,
				token,
				outputState,
				outputState,
				logits
			) == false ) {
				throw new Error("RWKV unexpected eval failed");
			}
		}

		// Return the output state
		return {
			state: outputState,
			logits: logits
		};
	}

	/**
	 * @private mehthod (do not use API directly, it will not be maintained)
	 * 
	 * Given the input string, get the hidden state buffer.
	 * Fetching it from the internal cache when possible.
	 * 
	 * Also updates the cache with a copy of the new state
	 * 
	 * @param {String} input string to get the hidden state for
	 * 
	 * @returns {Object} the hidden state buffer and logits buffer, along with input string and tokens
	 */
	_getHiddenState_fromFullInputString(input) {
		// Existing cached state obj
		let cachedState = null;

		// Throw if the input is empty
		if( input == null || input.length == 0 ) {
			throw new Error("RWKV input string is empty (are you missing a prompt?)");
		}

		// Try to get matching existing state from the cache first
		// ---
		if( this._stateCache ) {
			// Get all the cache keys
			let keys = [];
			for(const key of this._stateCache.keys()) {
				keys.push(key);
			}

			// Sort the keys by length, longest first
			keys = keys.sort((a, b) => b.length - a.length);

			// Loop over the keys, see if we can find a match
			for( const prefixKey of keys ) {
				if( input.startsWith(prefixKey) ) {
					// Found a matching key, we continue from there
					cachedState = this._stateCache.get(prefixKey);

					// Check if the get operation was successful
					if( cachedState ) {
						// We found a match, break out of the loop
						break;
					}
				}
			}
		}

		// Get the starting hidden state, the string it was used to compute it
		// and the remaining string to compute
		// ---
		let initState = null;
		let initInput = "";
		let initTokens = [];
		let remainingInput = input;

		// Check if we found a cached state
		if( cachedState ) {
			// We found a cached state, we continue from there
			initState = cachedState.state;
			initInput = cachedState.prompt;
			initTokens = cachedState.tokens;

			// Remove the cached state from the remaining input
			remainingInput = remainingInput.slice(initInput.length);
		}

		// Remaining tokens is empty, we can return the cached state
		if( remainingInput.length == 0 ) {
			return {
				state: cachedState.state,
				logits: cachedState.logits,
				prompt: input,
				tokens: cachedState.tokens,
				cachedTokenSize: cachedState.tokens.length
			};
		}

		// Convert the remaining input into tokens
		let remainingTokens = tokenizer.encode(remainingInput);
		let fullTokenArr = initTokens.concat(remainingTokens);

		// If remaining token count <= TOKEN_CACHE_OFFSET
		// it means that its ineligible for caching
		//
		// Alternatively if _stateCache is disabled
		//
		// So just do a simple compute and return
		// ---
		if( remainingTokens.length <= TOKEN_CACHE_OFFSET || this._stateCache == null ) {
			// Compute the hidden state from the existing state and the remaining tokens
			let ans = this._getHiddenState_fromExistingState_andTokenArr(initState, remainingTokens);

			// Return full state obj
			return {
				state: ans.state,
				logits: ans.logits,
				prompt: input,
				tokens: fullTokenArr,
				cachedTokenSize: cachedState? cachedState.tokens.length : 0
			}
		}

		// The request is eligible for caching, we compute the hidden state
		// into two parts, the cachable part and the non-cachable part
		//
		// Also we can assume caching is enabled
		// ---

		// Get the cachable tokens, and the non-cachable tokens
		let remaining_cachableTokens = remainingTokens.slice(0, remainingTokens.length - TOKEN_CACHE_OFFSET);
		let remaining_nonCachableTokens = remainingTokens.slice(remainingTokens.length - TOKEN_CACHE_OFFSET);

		// Compute the cachable hidden state from the existing state and the cachable tokens
		let cachableState = this._getHiddenState_fromExistingState_andTokenArr(initState, remaining_cachableTokens);

		// And its associated values
		let remaining_cachableTokens_str = tokenizer.decode(remaining_cachableTokens);
		let cachableStr = initInput + remaining_cachableTokens_str;
		let cachableTokens = initTokens.concat(remaining_cachableTokens);

		// Lets store the cachable state into the cache
		this._stateCache.set(cachableStr, {
			state: cachableState.state,
			logits: cachableState.logits,
			prompt: cachableStr,
			tokens: cachableTokens
		});

		// Compute the non-cachable hidden state from the existing state and the non-cachable tokens
		let nonCachableState = this._getHiddenState_fromExistingState_andTokenArr(cachableState.state, remaining_nonCachableTokens);

		// Return the full state obj
		return {
			state: nonCachableState.state,
			logits: nonCachableState.logits,
			prompt: input,
			tokens: fullTokenArr,
			cachedTokenSize: cachedState? cachedState.tokens.length : 0
		};
	}

	//-------------------------
	// Completion operation
	//-------------------------

	/**
	 * Perform the completion operation on the given input string, and various additional options
	 * 
	 * The following options are supported (same behavior as the OpenAI API):
	 * - prompt (String)     : the prompt to use for completion, if used with hidden state, prompt will be appended to existing state
	 * - max_tokens (Number) : the maximum number of tokens to generate       (default: 64)
	 * - temperature (Number): the temperature to use for generation          (default: 1.0)
	 * - top_p (Number)      : the top_p to use for generation                (default: 1.0)
	 * - stop (Array<String>): the stop sequence string to use for generation (default: [])
	 * 
	 * The following are addtionally supported options:
	 * - streamCallback (Function): the callback to use for streaming results
	 * - initState (Float32Array) : the initial hidden state to start generation from, if this is used, the internal cache will be ignored
	 * 
	 * @param {Object} options the options to use for completion, if given a string, this will be used as the prompt
	 */
	completion(opt) {
		// ctx safety
		this._checkContext();

		// Normalize string opt
		if( opt instanceof String || typeof opt == "string" ) {
			opt = {
				prompt: opt
			}
		}

		// Check if we have a prompt, if not, throw an error
		if( opt.prompt == null || opt.prompt.length == 0 ) {
			throw new Error("Prompt is required for completion operation");
		}

		// The prompt state obj to use (after processing the prompt)
		let promptStartState = null;

		// Start the timer
		let startTime = Date.now();

		// Get the starting state
		if( opt.initState ) {
			let promptTokens = tokenizer.encode(opt.prompt);
			promptStartState = this._getHiddenState_fromExistingState_andTokenArr(opt.initState, promptTokens);

			// // Include the tracked input prompt, and tokens
			// // to keep the object consistent with the other code path
			// promptStartState.prompt = opt.prompt;
			// promptStartState.tokens = promptTokens;
		} else {
			promptStartState = this._getHiddenState_fromFullInputString(opt.prompt);
		}

		// Propmpt completion timer
		let promptCompletionTime = Date.now()

		// Get the stop sequence longest string length
		let stopArr = opt.stop || [];
		if( stopArr instanceof String || typeof stopArr == "string" ) {
			stopArr = [stopArr];
		}

		// Maximum length of the stop sequence
		let stopSeqMaxLen = 0;
		for(const stopStr of stopArr) {
			stopSeqMaxLen = Math.max(stopSeqMaxLen, stopStr.length);
		}
		
		// The output string
		let outputStr = "";
		let outputTokens = [];

		// The streamed position
		let outputStream = opt.streamCallback;
		let streamPos = 0;

		// The stop sequence which as matched (if any)
		let stopSeqMatched = null;

		// Utility function, to format the output object
		function formatOutputObject() {
			// Get the output completion time
			let completionTime = Date.now();

			// The final output str
			let finalOutputStr = outputStr;

			// Prepare final output string, if stop sequence was matched, we remove it from the output
			if( stopSeqMatched ) {
				let lastIndex = outputStr.lastIndexOf(stopSeqMatched);
				finalOutputStr = outputStr.substring(0, lastIndex);
			}

			// Handle output streaming
			if( outputStream ) {
				let streamLimit = finalOutputStr.length;
				if( streamLimit > 0 && streamLimit > streamPos ) {
					outputStream(outputStr.slice(streamPos, streamLimit));
					streamPos = streamLimit;
				}
			}

			// Get the final timings
			let promptDuration = promptCompletionTime - startTime;
			let completionDuration = completionTime - promptCompletionTime;
			let totalDuration = completionTime - startTime;

			// Lets return with the prebuilt state and logits
			let ret = {
				// The prompt used
				prompt: promptStartState.prompt,

				// The completion string
				completion: finalOutputStr,

				// // Last RWKV internal state
				// // useful for dev / debugging
				// //
				// // Dropped because it is confusing (it contains the n-1 state typically)
				// rwkv: {
				// 	state: promptStartState.state,
				// 	logits: promptStartState.logits,
				// }

				// Usage tracking
				usage: {
					promptTokens: promptStartState.tokens.length,
					completionTokens: outputTokens.length,
					totalTokens: promptStartState.tokens.length + outputTokens.length,
					promptTokensCached: promptStartState.cachedTokenSize || 0,
				},

				// Performance timings
				perf: {
					// Get timings in ms
					promptTime: promptDuration,
					completionTime: completionDuration,
					totalTime: totalDuration,

					// Time per token
					timePerPrompt: promptDuration / (promptStartState.tokens.length - promptStartState.cachedTokenSize),
					timePerCompletion: completionDuration / outputTokens.length,
					timePerFullPrompt: promptDuration / promptStartState.tokens.length,
				}
			}

			// Get the tokens per second
			ret.perf.promptPerSecond = 1000.0 / ret.perf.timePerPrompt;
			ret.perf.completionPerSecond = 1000.0 / ret.perf.timePerCompletion;
			ret.perf.fullPromptPerSecond = 1000.0 / ret.perf.timePerFullPrompt;
			// console.log(ret);

			// Return the object
			return ret;
		}

		// Special handling of max_tokens = 0
		// which is used for prompt caching
		if( opt.max_tokens === 0 ) {
			return formatOutputObject();
		}

		// Lets start preparing a circular state buffer
		const BUFFER_SIZE = TOKEN_CACHE_OFFSET + 1;
		let stateBuffer = [];
		stateBuffer.length = BUFFER_SIZE;
		stateBuffer[0] = promptStartState;

		// Lets prepopulate the state buffer
		for(let i=1; i<BUFFER_SIZE; i++) {
			stateBuffer[i] = {
				state: new Float32Array(this._state_size),
				logits: new Float32Array(this._logits_size),
			}
		}

		// Get the max token count
		let maxTokens = opt.max_tokens || 64;

		// Temperature and top_p settings
		let temperature = opt.temperature || 1.0;
		let top_p = opt.top_p || 1.0;

		// !!! 1st token generation
		// ---

		// Lets generate the first token
		let curTokenObj = ai_utils.sampleLogits(stateBuffer[0].logits, temperature, top_p);

		// Store the current output state
		stateBuffer[0].outputStr = outputStr.slice();
		stateBuffer[0].outputTokens = outputTokens.slice();

		// And update the output string and token array
		outputTokens.push(curTokenObj.token);
		outputStr += tokenizer.decode([curTokenObj.token]);

		// Subsequent token generation
		// ---

		// The current buffer index
		let curBufferIndex = 0;
		let nxtBufferIndex = 1;

		// !!! main token gen loop
		// ---

		// Lets loop until we hit the max token count
		// or one of the several stop sequence
		for(let i=1; i<maxTokens; i++) {
			// Get the current state
			let curState = stateBuffer[curBufferIndex].state;
			// let curLogits = stateBuffer[curBufferIndex].logits;

			// Get the target next state
			let nxtState = stateBuffer[nxtBufferIndex].state;
			let nxtLogits = stateBuffer[nxtBufferIndex].logits;

			// Compute the next state
			let evalRes = cpp_bind.rwkv_eval(this._ctx, curTokenObj.token, curState, nxtState, nxtLogits);

			// Abort if eval failed
			if( evalRes == false ) {
				throw new Error("Unexpected eval error during inference process");
			}

			// Compute the token to sample with the given logits settings
			curTokenObj = ai_utils.sampleLogits(nxtLogits, temperature, top_p);

			// Store it in the state buffer the current output state
			stateBuffer[nxtBufferIndex].outputStr = outputStr.slice();
			stateBuffer[nxtBufferIndex].outputTokens = outputTokens.slice();

			// And update the output string and token array
			let curTokenStr = tokenizer.decode([curTokenObj.token]);
			outputTokens.push(curTokenObj.token);
			outputStr += curTokenStr;

			// Increment the buffer indexes
			curBufferIndex = nxtBufferIndex
			nxtBufferIndex++;
			if( nxtBufferIndex >= BUFFER_SIZE ) {
				nxtBufferIndex = 0;
			}

			// Check if we hit a stop sequence
			if( stopArr && stopArr.length > 0 ) {
				// Get the last X string, for stop sequence matching
				let lastXStr = outputStr.slice(-(stopSeqMaxLen*2));

				// Check if any of the stop sequences match
				for(const stopSeq of stopArr) {
					if( lastXStr.indexOf(stopSeq) >= 0 ) {
						stopSeqMatched = stopSeq;
						break;
					}
				}

				if(stopSeqMatched) {
					break;
				}
			}

			// Handle output streaming
			if( outputStream ) {
				let streamLimit = outputStr.length - stopSeqMaxLen*2;
				if( streamLimit > 0 && streamLimit > streamPos ) {
					outputStream(outputStr.slice(streamPos, streamLimit));
					streamPos = streamLimit;
				}
			}
		}


		// !!! finished the gen loop
		// ---

		// Check if its eligible for prompt caching
		if( this._stateCache == null || opt.initState != null ) {
			// This is not eligible for prompt caching
			// as either state cache is not enabled
			// or an initial state was provided
		} else if( outputTokens.length >= MIN_OUTPUT_TOKEN_SIZE_FOR_BUFFERING ) {
			// Store it into cache, only if there is sufficent distance from the previous cache entry
			// ---

			// Get the position of curBufferIndex - TOKEN_CACHE_OFFSET
			let cacheBufferIndex = curBufferIndex - TOKEN_CACHE_OFFSET - 1;
			if( cacheBufferIndex < 0 ) {
				cacheBufferIndex += BUFFER_SIZE;
			}

			// Get the cache buffer
			let cacheState = stateBuffer[cacheBufferIndex];

			// And store it into cache
			let cacheStr = promptStartState.prompt + cacheState.outputStr;
			this._stateCache[cacheStr] = {
				state: cacheState.state,
				logits: cacheState.logits,
				prompt: cacheStr,
				tokens: [].concat(promptStartState.tokens, cacheState.outputTokens),
			};
		}

		// Return the result
		return formatOutputObject();
	}

	/**
	 * Alias to `completion({ prompt:prompt, max_tokens:0 })`
	 * This is used to preload instruction set prompts into the cache, for later (faster) reuse
	 * 
	 * @param {String} prompt 
	 */
	preloadPrompt(prompt) {
		this.completion({ prompt:prompt, max_tokens:0 });
	}
}

// Provide a way for folks to use the direct CPP bind
RWKV.cpp_bind = cpp_bind;

//---------------------------
// Module export
//---------------------------
module.exports = RWKV;

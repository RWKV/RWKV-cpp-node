//---------------------------
// Dependencies
//---------------------------

const fs = require("fs")
const os = require("os");
const tokenizer = require("rwkv-tokenizer-node");
const cpp_bind = require("./cpp_bind").promises;
const ai_utils = require("./ai_utils");
const LRUCache = require("lru-cache");
const promise_queue = require("promise-queue");

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
	 * - path:           the model path
	 * 
	 * - threads:        the number of threads to use 
	 *                   (defaults to half the number of vCPUs)
	 * 
	 * - gpuOffload:     either the number of layers to offload, or a string ending with %
	 *                   used to indicate the % of layers of offload from the model
	 * 
	 * - concurrent:     the number of concurrent inferences to allow at a time
	 * 	                 (defaults to 1)
	 * 
	 * - batchSize:      the batch size to use for input inference handling
	 *                   (defaults to 64)
	 * 
	 * - stateCacheSize: the hidden state cache size to use,
	 *                   useful to speed up inference of a chat like model
	 *                   (defaults to 50)
	 *
	 * @param {Object|String} config object, or the model path string
	 */
	constructor(config) {
		// Check if the config is a string, if so normalize it to a config obj
		if (typeof config === "string") {
			config = {
				path: config
			};
		}

		// Get the CPU thread count
		let threads = config.threads;
		if (config.threads == null || threads <= 0) {
			if( config.gpuOffload != null && parseInt(config.gpuOffload) > 0) {
				// With gpu offloading, the optimal seems to be a light mix of cpu
				config.threads = 4;
			} else {
				// Use half the number of vCPUs
				config.threads = os.cpus().length / 2;
			}
		}

		// Store the used config
		this._config = config;
	}

	/**
	 * Setup the RWKV context
	 */
	async setup() {
		// Get the config object
		let config = this._config;

		// Get the file size 
		let fileStat = await fs.promises.stat(config.path);
		if( fileStat.isFile() == false ) {
			throw new Error("RWKV model path is not a file: " + config.path);
		}
		let fileSize = fileStat.size;
		this._fileSize = fileSize;
		
		// Load the cpp context, and store it
		let mainCtx = await cpp_bind.rwkv_init_from_file(config.path, config.threads);
		this._mainCtx = mainCtx;

		// Get the state and logits size
		this._state_size = await cpp_bind.rwkv_get_state_len(mainCtx);
		this._logits_size = await cpp_bind.rwkv_get_logits_len(mainCtx);

		// Offload layers
		let gpu_layers = config.gpuOffload || 0;
		if( gpu_layers.toString().endsWith("%") ) {
			// Get the int value
			let gpu_layers_percent = parseInt(gpu_layers);

			// Get the number of layers
			let num_layers = await cpp_bind.rwkv_get_n_layer(mainCtx);

			// Compute the number of layers to offload
			gpu_layers = Math.floor(num_layers * gpu_layers_percent / 100);
		}

		// GPU offloading if needed
		if (gpu_layers > 0) {
			await cpp_bind.rwkv_gpu_offload_layers(mainCtx, gpu_layers);
		}

		// Get the number of concurrent inferences
		let concurrent = config.concurrent || 1;

		// Prepare the work ctx array, used for seperate concurrent inferences
		let workCtxArray = [mainCtx];
		for(let i=1; i<concurrent; i++) {
			workCtxArray.push(await cpp_bind.rwkv_clone_context(mainCtx, config.threads));
		}
		this._workCtxArray = workCtxArray;

		// Setup the promise queues & the shared work queue
		// how this work is that inference task first goes into the shared queue,
		// and gets split into their respective worker queues.
		//
		// This is done to prevent any single worker queue from pilling up with
		// too many 'large tasks', and cause a misbalance in task distribution.
		let workQueueArr = [];
		for(let i=0; i<concurrent; i++) {
			workQueueArr.push(new promise_queue(1));
		}
		this._workQueueArr = workQueueArr;
		this._sharedWorkQueue = new promise_queue(concurrent * 2);

		// Prepare the LRU cache with the configured cache size
		//
		// it is worth noting that the 7B model takes up about 2.7 MB for the state buffer,
		// meaning you will need atleast 270 MB of RAM for a cachesize of 100
		//
		// The following are the values stored in the cache
		//
		// [prompt] -> {
		//	state: [state buffer],
		//	logits: [logits buffer],
		//	prompt: [prompt string],
		//	tokens: [token array]
		// }
		// ---
		// State cache, keeps the last N states in memory
		if (config.stateCacheSize === false || config.stateCacheSize <= 0) {
			// Disable the cache
			this._stateCache = null;
		} else {
			// Create the cache
			this._stateCache = new LRUCache({
				max: config.stateCacheSize || 50,
			});
		}
	}

	/**
	 * Cleanup the RWKV context - do not perform concurrent call of this operation
	 */
	async free() {
		// Destroy the context array
		if(this._workCtxArray) {
			for(let i=this._workCtxArray.length - 1; i>=1; i--) {
				await cpp_bind.rwkv_free(this._workCtxArray[i]);
			}
			this._workCtxArray = null;
		}
		// Destroy the main context
		if (this._mainCtx) {
			let p = await cpp_bind.rwkv_free(this._mainCtx);
			this._mainCtx = null;
			await p;
		}
	}

	//-------------
	// Queue handling ops
	//-------------

	/**
	 * @private mehthod (do not use API directly, it will not be maintained)
	 * 
	 * For the given async function, assign it to a worker, with the shortest queue.
	 * The function is only invoked when the worker is ready to accept the function.
	 * 
	 * This is used internally to distribute requests across worker contexts.
	 * While limiting execution to only 1 per worker.
	 * 
	 * @param {Function} func - the async function to assign
	 */
	async _assignFunctionToWorker(func) {
		// self ref
		let self = this;

		// First lets join the shared queue
		return await this._sharedWorkQueue.add(async () => {
			// Get the worker with the shortest queue, for us to assign the function to
			// ---
			let shortestQueue = this._workQueueArr[0];
			let shortestQueueIdx = 0;
			let shortestQueueLength = shortestQueue.getPendingLength() + shortestQueue.getQueueLength();
			for(let i=1; i<this._workQueueArr.length; i++) {
				let candidateQueue = this._workQueueArr[i];
				let candidateQueueLength = candidateQueue.getPendingLength() + candidateQueue.getQueueLength();
				if( candidateQueueLength < shortestQueueLength ) {
					shortestQueue = candidateQueue;
					shortestQueueIdx = i;
				}
			}

			// Join the worker specific queue
			return await shortestQueue.add(async () => {
				// Invoke the function, with worker instance, and the queue index
				return await func(self._workCtxArray[shortestQueueIdx], shortestQueueIdx);
			});
		});
	}

	/**
	 * The number of active worker processes
	 * @returns {number} the number of active workers
	 */
	activeWorkCount() {
		return this._sharedWorkQueue.getPendingLength();
	}

	/**
	 * The number of pending work requests
	 * @returns {number} the number of pending work requests
	 */
	pendingWorkCount() {
		return this._sharedWorkQueue.getQueueLength();
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
	 * @param {Float32Array} inState - existing state to compute from
	 * @param {Array<number>} tokenArr - array of tokens to compute
	 * @param {*} workerCtx - the worker context to use
	 *
	 * @returns {Object} the hidden state buffer and logits buffer
	 */
	async _getHiddenState_fromExistingState_andTokenArr(inState, tokenArr, workerCtx) {
		// Edge case handling when tokenArr is empty
		if (tokenArr == null || tokenArr.length == 0) {
			throw new Error("RWKV token array is empty");
		}
		if( workerCtx == null ) {
			throw new Error("RWKV worker context is null");
		}

		// Get the batch size
		let batchSize = this._config.batchSize || 1;

		// Prepare the output state to use
		let outputState = new Float32Array(this._state_size);

		// Copy the input state into the output state
		if (inState != null) {
			outputState.set(inState);
		}

		// Prepare the logit buffer (can be ignored safely)
		let logits = new Float32Array(this._logits_size);

		// Compute the hidden state for each token
		for (let i = 0; i < tokenArr.length; i += batchSize) {
			const chunk = tokenArr.slice(i, i + batchSize);
			if (
				(await cpp_bind.rwkv_eval_sequence(
					workerCtx,
					chunk,
					chunk.length,
					outputState,
					outputState,
					logits
				)) == false
			) {
				throw new Error("RWKV unexpected eval failed");
			}
		}

		// Return the output state
		return {
			state: outputState,
			logits: logits,
		};
	}

	/**
	 * Internal function, throw if the context is not set
	 */
	_checkContext() {
		if (!this._mainCtx) {
			throw new Error("RWKV context is not set, have you called setup()?");
		}
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
	 * @param {Array<number>} inputTokens - array of tokens to compute
	 * @param {*} workerCtx - the worker context to use
	 *
	 * @returns {Object} the hidden state buffer and logits buffer, along with input string and tokens
	 */
	async _getHiddenState_fromFullInputString(input, inputTokens, workerCtx) {
		// Existing cached state obj
		let cachedState = null;

		// Throw if the input is empty
		if (input == null || input.length == 0) {
			throw new Error("RWKV input string is empty (are you missing a prompt?)");
		}

		// Try to get matching existing state from the cache first
		// ---
		if (this._stateCache) {
			// Get all the cache keys
			let keys = [];
			for (const key of this._stateCache.keys()) {
				keys.push(key);
			}

			// Sort the keys by length, longest first
			keys = keys.sort((a, b) => b.length - a.length);

			// Loop over the keys, see if we can find a match
			for (const prefixKey of keys) {
				if (input.startsWith(prefixKey)) {
					// Found a matching key, lets proceed to check the tokens
					let candidateState = this._stateCache.get(prefixKey);
					if( candidateState == null ) {
						// candidate state was evicted from cache, skip
						continue;
					}

					// Check if the token array matches
					let inputTokensSlice = inputTokens.slice(candidateState.tokens.length);
					for( let i=0; i<inputTokensSlice.length; i++ ) {
						if( inputTokensSlice[i] != candidateState.tokens[i] ) {
							// Token mismatch, we continue, to next candidate
							continue;
						}
					}

					// We found a matching state, we break from here
					// Convert candidateState to cachedState
					cachedState = candidateState;
					break;
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
		if (cachedState) {
			// We found a cached state, we continue from there
			initState = cachedState.state;
			initInput = cachedState.prompt;
			initTokens = cachedState.tokens;

			// Remove the cached state from the remaining input
			remainingInput = remainingInput.slice(initInput.length);
		}

		// Remaining tokens is empty, we can return the cached state
		if (remainingInput.length == 0) {
			return {
				state: cachedState.state,
				logits: cachedState.logits,
				prompt: input,
				tokens: cachedState.tokens,
				cachedTokenSize: cachedState.tokens.length,
			};
		}

		// Convert the remaining input into tokens
		let remainingTokens = tokenizer.encode(remainingInput);
		let fullTokenArr = initTokens.concat(remainingTokens);

		// Compute the cachable hidden state from the existing state and the cachable tokens
		let finalState = await this._getHiddenState_fromExistingState_andTokenArr(
			initState,
			remainingTokens,
			workerCtx
		);

		// Lets store the result in the cache
		this._stateCache.set(input, {
			state: finalState.state,
			logits: finalState.logits,
			prompt: input,
			tokens: fullTokenArr,
		});

		// Return the full state obj
		return {
			state: finalState.state,
			logits: finalState.logits,
			prompt: input,
			tokens: fullTokenArr,
			cachedTokenSize: cachedState ? cachedState.tokens.length : 0,
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
	async completion(opt) {
		// mainCtx safety
		this._checkContext();

		// Self ref
		let self = this;

		// Normalize string opt
		if (opt instanceof String || typeof opt == "string") {
			opt = {
				prompt: opt,
			};
		}

		// Check if we have a prompt, if not, throw an error
		if (opt.prompt == null || opt.prompt.length == 0) {
			throw new Error("Prompt is required for completion operation");
		}

		// The request time
		let requestTime = Date.now();

		// ===
		// Perform the main completion operation computation within
		// a worker queue context, ensuring no overlapped executions of workers
		// ===
		return await this._assignFunctionToWorker(async(workerCtx) => {

			// ---
			// We first get the initial state from the prompt input
			// ---

			// Start the timer
			let startTime = Date.now();

			// The prompt state obj to use (after processing the prompt)
			let promptStartState = null;

			// Convert the prompt into tokens
			let promptTokens = tokenizer.encode(opt.prompt);

			// Get the starting state
			if (opt.initState) {
				// This skips the cache, as we are using an existing state
				promptStartState = await self._getHiddenState_fromExistingState_andTokenArr(
					opt.initState, promptTokens, workerCtx
				);
			} else {
				promptStartState = await self._getHiddenState_fromFullInputString(
					opt.prompt, promptTokens, workerCtx
				);
			}

			// Propmpt completion timer
			let promptCompletionTime = Date.now();

			// ---
			// Initialize various local vars
			// ---

			// Get the stop sequence longest string length
			let stopArr = opt.stop || [];
			if (stopArr instanceof String || typeof stopArr == "string") {
				stopArr = [stopArr];
			}

			// Maximum length of the stop sequence
			let stopSeqMaxLen = 0;
			for (const stopStr of stopArr) {
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

			// ---
			// Handle immediate output, for max_tokens = 0
			// which is used to cache the prompt for later use
			// ---

			// Utility function, to format the output object
			function formatOutputObject() {
				// Get the output completion time
				let completionTime = Date.now();

				// The final output str
				let finalOutputStr = outputStr;

				// Prepare final output string, if stop sequence was matched, we remove it from the output
				if (stopSeqMatched) {
					let lastIndex = outputStr.lastIndexOf(stopSeqMatched);
					finalOutputStr = outputStr.substring(0, lastIndex);
				}

				// Handle output streaming
				if (outputStream) {
					let streamLimit = finalOutputStr.length;
					if (streamLimit > 0 && streamLimit > streamPos) {
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

					// Usage tracking
					usage: {
						promptTokens: promptStartState.tokens.length,
						completionTokens: outputTokens.length,
						totalTokens: promptStartState.tokens.length + outputTokens.length,
						promptTokensCached: promptStartState.cachedTokenSize || 0,
					},

					// Performance timings
					perf: {
						// Idle queue waiting time (within queue)
						queueIdleTime: startTime - requestTime,
						
						// Get timings in ms
						promptTime: promptDuration,
						completionTime: completionDuration,
						totalTime: totalDuration,

						// Time per token
						timePerPrompt:
							promptDuration /
							(promptStartState.tokens.length - promptStartState.cachedTokenSize),
						timePerCompletion: completionDuration / outputTokens.length,
						timePerFullPrompt: promptDuration / promptStartState.tokens.length,
					},
				};

				// Get the tokens per second
				ret.perf.promptPerSecond = 1000.0 / ret.perf.timePerPrompt;
				ret.perf.completionPerSecond = 1000.0 / ret.perf.timePerCompletion;
				ret.perf.fullPromptPerSecond = 1000.0 / ret.perf.timePerFullPrompt;

				// Return the object
				return ret;
			}

			// Special handling of max_tokens = 0
			// which is used for prompt caching
			if (opt.max_tokens === 0) {
				return formatOutputObject();
			}

			// Lets start preparing a circular state buffer
			const BUFFER_SIZE = 5;
			let stateBuffer = [];
			stateBuffer.length = BUFFER_SIZE;
			stateBuffer[0] = promptStartState;

			// ---
			// Initialize circular state buffer
			// ---

			// Lets prepopulate the state buffer
			for (let i = 1; i < BUFFER_SIZE; i++) {
				stateBuffer[i] = {
					state: new Float32Array(self._state_size),
					logits: new Float32Array(self._logits_size),
				};
			}

			// Get the max token count
			let maxTokens = opt.max_tokens || 64;

			// Temperature and top_p settings
			let temperature = opt.temperature || 1.0;
			let top_p = opt.top_p || 1.0;

			// ---
			// !!! 1st token generation
			// ---

			// Lets generate the first token
			let curTokenObj = ai_utils.sampleLogits(
				stateBuffer[0].logits,
				temperature,
				top_p
			);

			// Store the current output state
			stateBuffer[0].outputStr = outputStr.slice();
			stateBuffer[0].outputTokens = outputTokens.slice();

			// And update the output string and token array
			outputTokens.push(curTokenObj.token);
			outputStr += tokenizer.decode([curTokenObj.token]);

			// The current buffer index
			let curBufferIndex = 0;
			let nxtBufferIndex = 1;

			// ---
			// !!! main token gen loop
			// ---

			// Lets loop until we hit the max token count
			// or one of the several stop sequence
			for (let i = 1; i < maxTokens; i++) {
				// Get the current state
				let curState = stateBuffer[curBufferIndex].state;
				// let curLogits = stateBuffer[curBufferIndex].logits;

				// Get the target next state
				let nxtState = stateBuffer[nxtBufferIndex].state;
				let nxtLogits = stateBuffer[nxtBufferIndex].logits;

				// Compute the next state
				let evalRes = await cpp_bind.rwkv_eval(
					workerCtx,
					curTokenObj.token,
					curState,
					nxtState,
					nxtLogits
				);
				//
				// Abort if eval failed
				if (evalRes == false) {
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
				curBufferIndex = nxtBufferIndex;
				nxtBufferIndex++;
				if (nxtBufferIndex >= BUFFER_SIZE) {
					nxtBufferIndex = 0;
				}

				// Check if we hit a stop sequence
				if (stopArr && stopArr.length > 0) {
					// Get the last X string, for stop sequence matching
					let lastXStr = outputStr.slice(-(stopSeqMaxLen * 2));

					// Check if any of the stop sequences match
					for (const stopSeq of stopArr) {
						if (lastXStr.indexOf(stopSeq) >= 0) {
							stopSeqMatched = stopSeq;
							break;
						}
					}

					if (stopSeqMatched) {
						break;
					}
				}

				// Handle output streaming
				if (outputStream) {
					let streamLimit = outputStr.length - stopSeqMaxLen * 2;
					if (streamLimit > 0 && streamLimit > streamPos) {
						outputStream(outputStr.slice(streamPos, streamLimit));
						streamPos = streamLimit;
					}
				}
			}

			// ---
			// !!! finished the gen loop
			// ---

			// Check if its eligible for prompt caching
			if (self._stateCache == null || opt.initState != null) {
				// This is not eligible for prompt caching
				// as either state cache is not enabled
				// or an initial state was provided
			} else {
				// Get the cache buffer
				let cacheState = stateBuffer[curBufferIndex];

				// And store it into cache
				let cacheStr = promptStartState.prompt + cacheState.outputStr;
				self._stateCache[cacheStr] = {
					state: cacheState.state,
					logits: cacheState.logits,
					prompt: cacheStr,
					tokens: [].concat(promptStartState.tokens, cacheState.outputTokens),
				};
			}

			// Return the result
			return formatOutputObject();
		});
	}

	/**
	 * Alias to `completion({ prompt:prompt, max_tokens:0 })`
	 * This is used to preload instruction set prompts into the cache, for later (faster) reuse
	 *
	 * @param {String} prompt
	 */
	async preloadPrompt(prompt) {
		await this.completion({ prompt: prompt, max_tokens: 0 });
	}
}

// Provide a way for folks to use the direct CPP bind
RWKV.cpp_bind = cpp_bind;

//---------------------------
// Module export
//---------------------------
module.exports = RWKV;

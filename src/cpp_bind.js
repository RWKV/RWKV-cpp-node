//---------------------------
// Dependencies
//---------------------------

// Node deps
const path = require("path")
const util = require("util")

// Get the koffi
const koffi = require("koffi");

//---------------------------
// Lib selection
//---------------------------

// The lib path to use
let rwkvCppLibPath = null;

// Check which platform we're on
if( process.arch === 'arm64' ) {
	if( process.platform === 'darwin' ) {
		rwkvCppLibPath = './lib/librwkv-arm64.dylib';
	} else if( process.platform === 'linux' ) {
		rwkvCppLibPath = './lib/librwkv-arm64.so';
	} else {
		throw new Error('Unsupported RWKV.cpp platform / arch: ' + process.platform + ' / ' + process.arch);
	}
} else if( process.arch === 'x64' ) {
	if( process.platform === 'win32' ) {
		// We only do CPU feature detection in windows
		// due to the different libraries with varients in AVX support
		//
		// Note as this is an optional dependency, 
		// it can fail to load/compile for random reasons
		let cpuFeatures = null;
		try {
			cpuFeatures = require('cpu-features')();
		} catch( err ) {
			// Silently ignore, we assume only avx is supported
		}
	
		// Load the highest AVX supported CPU when possible
		if( cpuFeatures == null ) {
			// console.warn("cpu-features failed to load, assuming AVX CPU is supported")
			rwkvCppLibPath = './lib/rwkv-avx.dll';
		} else if( cpuFeatures.flags.avx512 ) {
			rwkvCppLibPath = './lib/rwkv-avx512.dll';
		} else if( cpuFeatures.flags.avx2 ) {
			rwkvCppLibPath = './lib/rwkv-avx2.dll';
		} else {
			// AVX detection is not reliable, so if we fail to detect, we downgrade to lowest avx version
			rwkvCppLibPath = './lib/rwkv-avx.dll';
		}
	} else if( process.platform === 'darwin' ) {
		rwkvCppLibPath = './lib/librwkv.dylib';
	} else if( process.platform === 'linux' ) {
		rwkvCppLibPath = './lib/librwkv.so';
	} else {
		throw new Error('Unsupported RWKV.cpp platform / arch: ' + process.platform + ' / ' + process.arch);
	}	
} else {
  throw new Error("Unsupported RWKV.cpp arch: " + process.arch);
}
// The lib path to use
const rwkvCppFullLibPath = path.resolve(__dirname, "..", rwkvCppLibPath);

//---------------------------
// Lib binding loading
//---------------------------

const rwkvKoffiBind = koffi.load(rwkvCppFullLibPath);

// Custom pointers, to avoid copying data to JS land
const ctx_pointer = koffi.pointer('CTX_HANDLE', koffi.opaque());

// Initializing / cloning process
const rwkv_init_from_file = rwkvKoffiBind.func('CTX_HANDLE rwkv_init_from_file(const char * model_file_path, uint32_t n_threads)');
const rwkv_clone_context = rwkvKoffiBind.func('CTX_HANDLE rwkv_clone_context(CTX_HANDLE ctx, uint32_t n_threads)');
const rwkv_gpu_offload_layers = rwkvKoffiBind.func('bool rwkv_gpu_offload_layers(CTX_HANDLE ctx, uint32_t n_gpu_layers)');

// Model info extraction
const rwkv_get_n_vocab = rwkvKoffiBind.func('size_t rwkv_get_n_vocab(CTX_HANDLE ctx)'); 
const rwkv_get_n_embed = rwkvKoffiBind.func('size_t rwkv_get_n_embed(CTX_HANDLE ctx)');
const rwkv_get_n_layer = rwkvKoffiBind.func('size_t rwkv_get_n_layer(CTX_HANDLE ctx)');
const rwkv_get_state_len = rwkvKoffiBind.func('size_t rwkv_get_state_len(CTX_HANDLE ctx)');  
const rwkv_get_logits_len = rwkvKoffiBind.func('size_t rwkv_get_logits_len(CTX_HANDLE ctx)');

// Eval sequence
const rwkv_eval = rwkvKoffiBind.func('bool rwkv_eval(CTX_HANDLE ctx, int32_t token, const float * state_in, _Out_ float * state_out, _Out_ float * logits_out)');
const rwkv_eval_sequence = rwkvKoffiBind.func('bool rwkv_eval_sequence(CTX_HANDLE ctx, const uint32_t * tokens, size_t sequence_len, const float * state_in, _Out_ float * state_out, _Out_ float * logits_out)');

// // Unsupported functions (due to API integration limitation)
// const rwkv_init_state = rwkvKoffiBind.func('void rwkv_init_state(CTX_HANDLE ctx, float * state)'); 
// const rwkv_set_print_errors = rwkvKoffiBind.func('void rwkv_set_print_errors(CTX_HANDLE ctx, bool print_errors)');  
// const rwkv_get_print_errors = rwkvKoffiBind.func('bool rwkv_get_print_errors(CTX_HANDLE ctx)');  
// const rwkv_get_last_error = rwkvKoffiBind.func('enum rwkv_error_flags rwkv_get_last_error(CTX_HANDLE ctx)');
// const rwkv_get_system_info_string = rwkvKoffiBind.func('const char * rwkv_get_system_info_string()');

// Quantizing models
const rwkv_quantize_model_file = rwkvKoffiBind.func('bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name)');

// Context destruction
const rwkv_free = rwkvKoffiBind.func('void rwkv_free(CTX_HANDLE ctx)');

//---------------------------
// Module export
//---------------------------

module.exports = {
  // The path to the lib used
  _libPath: rwkvCppFullLibPath,

  /**
   * Loads the model from a file and prepares it for inference.
   * Returns NULL on any error. Error messages would be printed to stderr.
   *
   * @param {String} model_file_path path to model file in ggml format.
   * @param {Number} n_threads number of threads to use for inference.
   *
   * @returns {ffi_pointer} Pointer to the RWKV context.
   */
  async rwkv_init_from_file(model_file_path, n_threads) {
    return new Promise((resolve, reject) => {
      rwkv_init_from_file.async(
        model_file_path,
        n_threads,
        (err, ctx) => {
          if (err) {
            reject(err);
          } else {
            resolve(ctx);
          }
        }
      );
    });
  },
  

  /**
   * Offloads the specified layers to the GPU.
   * Returns false on any error. Error messages would be printed to stderr.
   *
   */
  async rwkv_gpu_offload_layers(ctx, gpu_id) {
    return new Promise((resolve, reject) => {
      rwkv_gpu_offload_layers.async(ctx, gpu_id, (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  },

  /**
   * Frees all allocated memory and the context.
   *
   * @param {ffi_pointer} ctx - Pointer to the RWKV context.
   **/
  async rwkv_free(ctx) {
    return new Promise((resolve, reject) => {
      rwkv_free.async(ctx, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  },

  /**
   * Evaluates the model for a single token.
   * Returns false on any error. Error messages would be printed to stderr.
   *
   * @param {ffi_pointer} ctx - Pointer to the RWKV context.
   * @param {Number} token - The token to evaluate.
   * @param {ffi_pointer} state_in - The input state.
   * @param {ffi_pointer} state_out - The output state.
   * @param {ffi_pointer} logits_out - The output logits.
   *
   * @returns {Boolean} True if successful, false if not.
   **/
  async rwkv_eval(ctx, token, state_in, state_out, logits_out) {
    return new Promise((resolve, reject) => {
      rwkv_eval.async(
        ctx,
        token,
        state_in,
        state_out,
        logits_out,
        (err, result) => {
          if (err) {
            reject(err);
          } else {
            resolve(result);
          }
        }
      );
    });
  },

  /**  Evaluates the model for a sequence of tokens.
   * Uses a faster algorithm than rwkv_eval if you do not need the state and logits for every token. Best used with batch sizes of 64 or so.
   * Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.
   * Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
   * Returns false on any error.
   * @param tokens: pointer to an array of tokens. If NULL, the graph will be built and cached, but not executed: this can be useful for initialization.
   * @param sequence_len: number of tokens to read from the array.
   * @param state_in: FP32 buffer of size rwkv_get_state_len(), or NULL if this is a first pass.
   * @param state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
   * @param logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
   **/
  async rwkv_eval_sequence(
    ctx,
    tokens,
    sequence_len,
    state_in,
    state_out,
    logits_out
  ) {
    return new Promise((resolve, reject) => {
      rwkv_eval_sequence.async(
        ctx,
        tokens,
        sequence_len,
        state_in,
        state_out,
        logits_out,
        (err, result) => {
          if (err) {
            reject(err);
          } else {
            resolve(result);
          }
        }
      );
    });
  },

  /**
   * Returns count of FP32 elements in state buffer.
   *
   * @param {ffi_pointer} ctx - Pointer to the RWKV context.
   *
   * @returns {Number} The number of elements in the state buffer.
   **/
  async rwkv_get_state_buffer_element_count(ctx) {
    return new Promise((resolve, reject) => {
      rwkv_get_state_buffer_element_count.async(ctx, (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  },

  /**
   * Returns count of FP32 elements in logits buffer.
   *
   * @param {ffi_pointer} ctx - Pointer to the RWKV context.
   *
   * @returns {Number} The number of elements in the logits buffer.
   **/
  async rwkv_get_logits_buffer_element_count(ctx) {
    return new Promise((resolve, reject) => {
      rwkv_get_logits_buffer_element_count.async(ctx, (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  },

  /**
   * Quantizes the model file.
   * Returns false on any error. Error messages would be printed to stderr.
   *
   * Available format names:
   * - Q4_0
   * - Q4_1
   * - Q4_2
   * - Q5_0
   * - Q5_1
   * - Q8_0
   *
   * @param {String} model_file_path_in - Path to the input model file in ggml format.
   * @param {String} model_file_path_out - Path to the output model file in ggml format.
   * @param {String} format_name - The format to use for quantization.
   *
   * @returns {Boolean} True if successful, false if not.
   **/
  async rwkv_quantize_model_file(
    model_file_path_in,
    model_file_path_out,
    format_name
  ) {
    return new Promise((resolve, reject) => {
      rwkv_quantize_model_file.async(
        model_file_path_in,
        model_file_path_out,
        format_name,
        (err, result) => {
          if (err) {
            reject(err);
          } else {
            resolve(result);
          }
        }
      );
    });
  },

	// The path to the lib used
	_libPath: rwkvCppFullLibPath,

	// Initializing / cloning process
	// ---

	/**
	 * Loads the model from a file and prepares it for inference.
	 * Returns NULL on any error. Error messages would be printed to stderr.
	 * 
	 * @param {String} model_file_path - path to model file in ggml format.
	 * @param {Number} n_threads - number of CPU threads to use for inference.
	 * 
	 * @returns {ffi_pointer} Pointer to the RWKV context.
	 */
	rwkv_init_from_file: rwkv_init_from_file,

	/**
	 * Creates a new context from an existing one.
	 * This can allow you to run multiple rwkv_eval's in parallel, without having to load a single model multiple times.
	 * Each rwkv_context can have one eval running at a time.
	 * Every rwkv_context must be freed using rwkv_free.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * @param {Number} n_threads - number of CPU threads to use for inference.
	 * 
	 * @returns {ffi_pointer} Pointer to the new RWKV context.
	 */
	rwkv_clone_context: rwkv_clone_context,

	/**
	 * Offloads the specified layers to the GPU.
	 * Returns false on any error. Error messages would be printed to stderr.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * @param {Number} n_gpu_layers - number of GPU layers to offload
	 */
	rwkv_gpu_offload_layers: rwkv_gpu_offload_layers,

	// Model info extraction
	// ---

	/**
	 * Returns count of FP32 elements in state buffer.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of elements in the state buffer.
	 **/
	rwkv_get_state_len: rwkv_get_state_len,

	/**
	 * Returns count of FP32 elements in logits buffer.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of elements in the logits buffer.
	 **/
	rwkv_get_logits_len: rwkv_get_logits_len,

	/**
	 * Returns count of FP32 number of layers
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of layers
	 **/
	rwkv_get_n_layer: rwkv_get_n_layer,

	/**
	 * Returns count of FP32 number of embed params
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of embed params
	 **/
	rwkv_get_n_embed: rwkv_get_n_embed,

	/**
	 * Returns count of FP32 number of vocab params
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of embed params
	 **/
	rwkv_get_n_vocab: rwkv_get_n_vocab,

	// Eval sequences
	// ---

	/**
	 * Evaluates the model for a single token.
	 * Returns false on any error. Error messages would be printed to stderr.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * @param {Number} token - The token to evaluate.
	 * @param {ffi_pointer} state_in - The input state.
	 * @param {ffi_pointer} state_out - The output state.
	 * @param {ffi_pointer} logits_out - The output logits.
	 * 
	 * @returns {Boolean} True if successful, false if not.
	 **/
	rwkv_eval: rwkv_eval,

	/** 
	 * Evaluates the model for a sequence of tokens.
	 * Uses a faster algorithm than rwkv_eval if you do not need the state and logits for every token. Best used with batch sizes of 64 or so.
	 * Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.
	 * Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
	 * Returns false on any error.
	 * 
	 * @param tokens: pointer to an array of tokens. If NULL, the graph will be built and cached, but not executed: this can be useful for initialization.
	 * @param sequence_len: number of tokens to read from the array.
	 * @param state_in: FP32 buffer of size rwkv_get_state_len(), or NULL if this is a first pass.
	 * @param state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
	 * @param logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
	 * 
	 * @returns {Boolean} True if successful, false if not.
	 **/
	rwkv_eval_sequence : rwkv_eval_sequence,

	// Quantizing models
	// ---

	/**
	 * Quantizes the model file.
	 * Returns false on any error. Error messages would be printed to stderr.
	 * 
	 * Available format names:
	 * - Q4_0
	 * - Q4_1
	 * - Q4_2
	 * - Q5_0
	 * - Q5_1
	 * - Q8_0
	 *
	 * (For async op, just call the <function-name>.async varient)
	 * 
	 * @param {String} model_file_path_in - Path to the input model file in ggml format.
	 * @param {String} model_file_path_out - Path to the output model file in ggml format.
	 * @param {String} format_name - The quantization format to use.
	 * 
	 * @returns {Boolean} True if successful, false if not.
	 **/
	rwkv_quantize_model_file: rwkv_quantize_model_file,

	// Context destruction
	// ---

	/**
	 * Frees all allocated memory and the context.
	 *
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 **/
	rwkv_free: rwkv_free,

	// ====
	// Promise Varient
	// ====
	promises: {

		/**
		 * @async
		 * 
		 * Loads the model from a file and prepares it for inference.
		 * Returns NULL on any error. Error messages would be printed to stderr.
		 * 
		 * @param {String} model_file_path - path to model file in ggml format.
		 * @param {Number} n_threads - number of CPU threads to use for inference.
		 * 
		 * @returns {ffi_pointer} Pointer to the RWKV context.
		 */
		rwkv_init_from_file: util.promisify(rwkv_init_from_file.async),


		/**
		 * @async
		 * 
		 * Creates a new context from an existing one.
		 * This can allow you to run multiple rwkv_eval's in parallel, without having to load a single model multiple times.
		 * Each rwkv_context can have one eval running at a time.
		 * Every rwkv_context must be freed using rwkv_free.
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * @param {Number} n_threads - number of CPU threads to use for inference.
		 * 
		 * @returns {ffi_pointer} Pointer to the new RWKV context.
		 */
		rwkv_clone_context: util.promisify(rwkv_clone_context.async),


		/**
		 * @async
		 * 
		 * Offloads the specified layers to the GPU.
		 * Returns false on any error. Error messages would be printed to stderr.
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * @param {Number} n_gpu_layers - number of GPU layers to offload
		 */
		rwkv_gpu_offload_layers: util.promisify(rwkv_gpu_offload_layers.async),

		/**
		 * @async
		 * 
		 * Returns count of FP32 elements in state buffer.
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * 
		 * @returns {Number} The number of elements in the state buffer.
		 **/
		rwkv_get_state_len: util.promisify(rwkv_get_state_len.async),

		/**
		 * @async
		 * 
		 * Returns count of FP32 elements in logits buffer.
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * 
		 * @returns {Number} The number of elements in the logits buffer.
		 **/
		rwkv_get_logits_len: util.promisify(rwkv_get_logits_len.async),

		/**
		 * @async
		 * 
		 * Returns count of FP32 number of layers
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * 
		 * @returns {Number} The number of layers
		 **/
		rwkv_get_n_layer: util.promisify(rwkv_get_n_layer.async),

		/**
		 * @async
		 * 
		 * Returns count of FP32 number of embed params
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * 
		 * @returns {Number} The number of embed params
		 **/
		rwkv_get_n_embed: util.promisify(rwkv_get_n_embed.async),

		/**
		 * @async
		 * 
		 * Returns count of FP32 number of vocab params
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * 
		 * @returns {Number} The number of embed params
		 **/
		rwkv_get_n_vocab: rwkv_get_n_vocab,

		/**
		 * @async
		 * 
		 * Evaluates the model for a single token.
		 * Returns false on any error. Error messages would be printed to stderr.
		 * 
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 * @param {Number} token - The token to evaluate.
		 * @param {ffi_pointer} state_in - The input state.
		 * @param {ffi_pointer} state_out - The output state.
		 * @param {ffi_pointer} logits_out - The output logits.
		 * 
		 * @returns {Boolean} True if successful, false if not.
		 **/
		rwkv_eval: util.promisify(rwkv_eval.async),

		/** 
		 * @async
		 * 
		 * Evaluates the model for a sequence of tokens.
		 * Uses a faster algorithm than rwkv_eval if you do not need the state and logits for every token. Best used with batch sizes of 64 or so.
		 * Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.
		 * Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
		 * Returns false on any error.
		 * 
		 * @param tokens: pointer to an array of tokens. If NULL, the graph will be built and cached, but not executed: this can be useful for initialization.
		 * @param sequence_len: number of tokens to read from the array.
		 * @param state_in: FP32 buffer of size rwkv_get_state_len(), or NULL if this is a first pass.
		 * @param state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
		 * @param logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
		 * 
		 * @returns {Boolean} True if successful, false if not.
		 **/
		rwkv_eval_sequence: util.promisify(rwkv_eval_sequence.async),

		/**
		 * @async
		 * 
		 * Quantizes the model file.
		 * Returns false on any error. Error messages would be printed to stderr.
		 * 
		 * Available format names:
		 * - Q4_0
		 * - Q4_1
		 * - Q4_2
		 * - Q5_0
		 * - Q5_1
		 * - Q8_0
		 *
		 * (For async op, just call the <function-name>.async varient)
		 * 
		 * @param {String} model_file_path_in - Path to the input model file in ggml format.
		 * @param {String} model_file_path_out - Path to the output model file in ggml format.
		 * @param {String} format_name - The quantization format to use.
		 * 
		 * @returns {Boolean} True if successful, false if not.
		 **/
		rwkv_quantize_model_file: util.promisify(rwkv_quantize_model_file.async),

		/**
		 * @async
		 * 
		 * Frees all allocated memory and the context.
		 *
		 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
		 **/
		rwkv_free: util.promisify(rwkv_free.async),

	}
}

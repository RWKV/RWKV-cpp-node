//---------------------------
// Dependencies
//---------------------------

// Node deps
const path = require("path")

// Get the koffi
const koffi = require("koffi")

//---------------------------
// Lib selection
//---------------------------

// The lib path to use
let rwkvCppLibPath = null;

// console.log("ARCH", process.arch);

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
		// Note as this is an optional dependency, it can fail to load
		let cpuFeatures = null;
		try {
			cpuFeatures = require('cpu-features')();
		} catch( err ) {
			// Silently ignore, we assume avx
		}
	
		// Load the highest AVX supported CPU when possible
		if( cpuFeatures == null ) {
			// console.warn("cpu-features failed to load, assuming AVX CPU is supported")
			rwkvCppLibPath = './lib/rwkv-avx.dll';
		} else if( cpuFeatures.avx512 ) {
			rwkvCppLibPath = './lib/rwkv-avx512.dll';
		} else if( cpuFeatures.avx2 ) {
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
	throw new Error('Unsupported RWKV.cpp arch: ' + process.arch);
}
// The lib path to use
const rwkvCppFullLibPath = path.resolve( __dirname, "..", rwkvCppLibPath);

//---------------------------
// Lib binding loading
//---------------------------

// // Setup the FFI RWKV binding
// const rwkvFFiBind = ffi.Library(
// 	// The library path
// 	rwkvCppFullLibPath,
// 	// The functions to bind
// 	{
// 		// rwkv_context * rwkv_init_from_file(const char * model_file_path, uint32_t n_threads);
// 		'rwkv_init_from_file': ['pointer', ['CString', 'uint32']],
// 		// void rwkv_free(struct rwkv_context * ctx);
// 		'rwkv_free': ['void', ['pointer']],
// 		// bool rwkv_eval(struct rwkv_context * ctx, int32_t token, float * state_in, float * state_out, float * logits_out);
// 		'rwkv_eval': ['bool', ['pointer', 'int32', 'pointer', 'pointer', 'pointer']],
// 		// uint32_t rwkv_get_state_buffer_element_count(struct rwkv_context * ctx);
// 		'rwkv_get_state_buffer_element_count': ['uint32', ['pointer']],
// 		// uint32_t rwkv_get_logits_buffer_element_count(struct rwkv_context * ctx);
// 		'rwkv_get_logits_buffer_element_count': ['uint32', ['pointer']],
// 		// bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name);
// 		'rwkv_quantize_model_file': ['bool', ['CString', 'CString', 'CString']],
// 		// const char * rwkv_get_system_info_string();
// 		'rwkv_get_system_info_string': ['CString', []],
// 	}
// )

const rwkvKoffiBind = koffi.load(rwkvCppFullLibPath);
const ctx_pointer = koffi.pointer('CTX_HANDLE', koffi.opaque());
const rwkv_init_from_file = rwkvKoffiBind.func('CTX_HANDLE rwkv_init_from_file(const char * model_file_path, uint32_t n_threads)');
const rwkv_free = rwkvKoffiBind.func('void rwkv_free(CTX_HANDLE ctx)');
const rwkv_eval = rwkvKoffiBind.func('bool rwkv_eval(CTX_HANDLE ctx, int32_t token, _Inout_ float * state_in, _Inout_ float * state_out, _Inout_ float * logits_out)');
const rwkv_get_state_buffer_element_count = rwkvKoffiBind.func('uint32_t rwkv_get_state_buffer_element_count(CTX_HANDLE)');
const rwkv_get_logits_buffer_element_count = rwkvKoffiBind.func('uint32_t rwkv_get_logits_buffer_element_count(CTX_HANDLE)');
const rwkv_gpu_offload_layers = rwkvKoffiBind.func('bool rwkv_gpu_offload_layers(CTX_HANDLE ctx, uint32_t gpu_id)');
const rwkv_eval_sequence = rwkvKoffiBind.func('bool rwkv_eval_sequence(CTX_HANDLE ctx, const uint32_t * tokens, size_t sequence_len, _Inout_ const float * state_in, _Inout_ float * state_out, _Inout_ float * logits_out)');
const rwkv_quantize_model_file = rwkvKoffiBind.func('bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name)');
const rwkv_get_system_info_string = rwkvKoffiBind.func('const char * rwkv_get_system_info_string()');

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
	rwkv_init_from_file: rwkv_init_from_file,

	/**
	 * Offloads the specified layers to the GPU.
	 * Returns false on any error. Error messages would be printed to stderr.
	 * 
	 */
	rwkv_gpu_offload_layers: rwkv_gpu_offload_layers,

	/**
	 * Frees all allocated memory and the context.
	 *
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 **/
	rwkv_free: rwkv_free,

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
    rwkv_eval_sequence : rwkv_eval_sequence,

	/**
	 * Returns count of FP32 elements in state buffer.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of elements in the state buffer.
	 **/
	rwkv_get_state_buffer_element_count: rwkv_get_state_buffer_element_count,


	/**
	 * Returns count of FP32 elements in logits buffer.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * 
	 * @returns {Number} The number of elements in the logits buffer.
	 **/
	rwkv_get_logits_buffer_element_count: rwkv_get_logits_buffer_element_count,

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
	 * @param {String} format_name - The quantization format to use.
	 * 
	 * @returns {Boolean} True if successful, false if not.
	 **/
	rwkv_quantize_model_file: rwkv_quantize_model_file,

	/**
	 * Returns a string with system information.
	 *
	 * @returns {String} The system information string.
	 **/
	rwkv_get_system_info_string: rwkv_get_system_info_string,


}
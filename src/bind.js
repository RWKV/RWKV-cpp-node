//---------------------------
// Dependencies
//---------------------------

// Node deps
const os = require("os")
const path = require("path")

// Get the FFI / NAPI bindings
const ffi = require('ffi-napi')

//---------------------------
// Lib selection
//---------------------------

// The lib path to use
let rwkvCppLibPath = null;

// Check which platform we're on
if( process.platform === 'win32' ) {
	// We only do CPU feature detection in windows
	// due to the different libraries with varients in AVX support
	const cpuFeatures = require('cpu-features')();

	// Load the highest AVX supported CPU when possible
	if( cpuFeatures.avx512 ) {
		rwkvCppLibPath = './lib/rwkv-avx512.dylib';
	} else if( cpuFeatures.avx2 ) {
		rwkvCppLibPath = './lib/rwkv-avx2.dylib';
	} else if( cpuFeatures.avx ) {
		rwkvCppLibPath = './lib/rwkv-avx.dylib';
	} else {
		throw new Error("Missing AVX CPU support (Require 2012 or newer intel/amd CPU, if running inside a VM ensure your CPU passthrough is host)")
	}
} else if( process.platform === 'darwin' ) {
	rwkvCppLibPath = './lib/librwkv.dylib';
} else if( process.platform === 'linux' ) {
	rwkvCppLibPath = './lib/librwkv.so';
} else {
	throw new Error('Unsupported RWKV.cpp platform: ' + process.platform);
}

//---------------------------
// Lib binding loading
//---------------------------

// Setup the RWKV binding
const rwkvFFiBind = ffi.Library(
	// The library path
	path.resolve(__dirname, rwkvCppLibPath),

	// The functions to bind
	{
		// rwkv_context * rwkv_init_from_file(const char * model_file_path, uint32_t n_threads);
		'rwkv_init_from_file': ['pointer', ['CString', 'uint32']],

		// void rwkv_free(struct rwkv_context * ctx);
		'rwkv_free': ['void', ['pointer']],

		// bool rwkv_eval(struct rwkv_context * ctx, int32_t token, float * state_in, float * state_out, float * logits_out);
		'rwkv_eval': ['bool', ['pointer', 'int32', 'pointer', 'pointer', 'pointer']],

		// uint32_t rwkv_get_state_buffer_element_count(struct rwkv_context * ctx);
		'rwkv_get_state_buffer_element_count': ['uint32', ['pointer']],

		// uint32_t rwkv_get_logits_buffer_element_count(struct rwkv_context * ctx);
		'rwkv_get_logits_buffer_element_count': ['uint32', ['pointer']],

		// bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name);
		'rwkv_quantize_model_file': ['bool', ['CString', 'CString', 'CString']],

		// const char * rwkv_get_system_info_string();
		'rwkv_get_system_info_string': ['CString', []],
	}
)

//---------------------------
// Module export
//---------------------------

module.exports = {
	
	/**
	 * Loads the model from a file and prepares it for inference.
	 * Returns NULL on any error. Error messages would be printed to stderr.
	 * 
	 * @param {String} model_file_path path to model file in ggml format.
	 * @param {Number} n_threads number of threads to use for inference.
	 * 
	 * @returns {ffi_pointer} Pointer to the RWKV context.
	 */
	rwkv_init_from_file: rwkvFFiBind.rwkv_init_from_file,

	/**
	 * Frees all allocated memory and the context.
	 *
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 **/
	rwkv_free: rwkvFFiBind.rwkv_free,

	/**
	 * Evaluates the model for a single token.
	 * Returns false on any error. Error messages would be printed to stderr.
	 * 
	 * @param {ffi_pointer} ctx - Pointer to the RWKV context.
	 * @param {Number} token - The token to evaluate.
	 * @param {ffi_pointer} state_in - The input state.
	 * @param {ffi_pointer} state_out - The output state.
	 * @param {ffi_pointer} logits_out - The output logits.
	 **/
	rwkv_eval: rwkvFFiBind.rwkv_eval,

	// /**
	//  * Returns count of FP32 elements in state buffer.
	//  *
}
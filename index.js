//---------------------------
// Dependencies
//---------------------------

// Node deps
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
// Lib selection
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





// // Variable types
// const uint32 = ref.types.uint32;
// const int32 = ref.types.int32;
// const bool = ref.types.bool;
// const double = ref.types.double;
// const float = ref.types.float;
// const char = ref.types.char;

// const RWKV_API = {
//   'win32': 'rwkv.dll',
//   'darwin': 'librwkv.dylib',
//   'linux': 'librwkv.so'
// }[process.platform];

// // Define the required functions with their C arguments and return types
// const rwkv_context = ref.types.void;
// const rwkv_contextPtr = ref.refType(rwkv_context);
// const stringPtr = ref.refType(char);
// const voidPtr = ref.refType(ref.types.void);
// const rwkv_init_from_file = ffi.Library(RWKV_API, {
//   'rwkv_init_from_file': [rwkv_contextPtr, [stringPtr, uint32]],
// });
// const rwkv_eval = ffi.Library(RWKV_API, {
//   'rwkv_eval': [bool, [rwkv_contextPtr, int32, float, float, float]],
// });
// const rwkv_get_state_buffer_element_count = ffi.Library(RWKV_API, {
//   'rwkv_get_state_buffer_element_count': [uint32, [rwkv_contextPtr]],
// });
// const rwkv_get_logits_buffer_element_count = ffi.Library(RWKV_API, {
//   'rwkv_get_logits_buffer_element_count': [uint32, [rwkv_contextPtr]],
// });
// const rwkv_free = ffi.Library(RWKV_API, {
//   'rwkv_free': ['void', [rwkv_contextPtr]],
// });
// const rwkv_quantize_model_file = ffi.Library(RWKV_API, {
//   'rwkv_quantize_model_file': [bool, [stringPtr, stringPtr, stringPtr]],
// });
// const rwkv_get_system_info_string = ffi.Library(RWKV_API, {
//   'rwkv_get_system_info_string': [stringPtr, []],
// });

// // Export the functions for usage in Node.js
// module.exports = {
//   rwkv_init_from_file,
//   rwkv_eval,
//   rwkv_get_state_buffer_element_count,
//   rwkv_get_logits_buffer_element_count,
//   rwkv_free,
//   rwkv_quantize_model_file,
//   rwkv_get_system_info_string
// };
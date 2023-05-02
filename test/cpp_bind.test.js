//-----------------------------------------------------------
// Setup chai assertion
//-----------------------------------------------------------
const chai = require("chai");
const chaiAsPromised = require("chai-as-promised");
chai.use(chaiAsPromised);
const expect = chai.expect;
const assert = chai.assert;

//-----------------------------------------------------------
// Load dependencies
//-----------------------------------------------------------

const os = require("os")
const cpp_bind = require("../src/cpp_bind")
const modelPath = "./raven/Q8_0-RWKV-4-Raven-7B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230430-ctx8192.bin"

//-----------------------------------------------------------
// And perform the unit tests
//-----------------------------------------------------------

// Validate the R1 action decider step
describe("Loading the RWKV model", function() {
	// Set large timeout
	this.timeout(60 * 1000); // 60 seconds

	// RWKV context
	let ctx = null;
	let state_size = 0;
	let logits_size = 0;

	// Lets load the model, and get a ctx pointer
	it("load the model", function() {
		// Lets load the model, tagged to the number of threads
		ctx = cpp_bind.rwkv_init_from_file(modelPath, os.cpus().length);

		// Validate the context is not null
		assert.ok(ctx != null, "Context Loaded check");
	});

	it("get the system info", function() {
		// Lets load the model, tagged to the number of threads
		let info = cpp_bind.rwkv_get_system_info_string();

		// Validate the system info is not null
		assert.ok(info != null, "System info NULL check");
		assert.ok(info.indexOf("AVX") >= 0, "AVX check (check for the key, not the result)");
	});

	// Get the state and logits size
	it("check state and logits size", function() {
		// Get the state and logits size
		state_size = cpp_bind.rwkv_get_state_buffer_element_count(ctx);
		logits_size = cpp_bind.rwkv_get_logits_buffer_element_count(ctx);

		// Validate the state and logits size, for 7B v11 model
		assert.equal(state_size, 655360, "State size check");
		assert.equal(logits_size, 50277, "Logit size check");
	});

	// Helper function, to create a state buffer
	function createStateBuffer() {
		return new Float32Array(state_size);
	}

	// Helper function, to create a logits buffer
	function createLogitsBuffer() {
		return new Float32Array(logits_size);
	}

	// Single pass test
	it("lets do a single pass", function() {
		// Initialize the state and logits buffer
		const in_state = createStateBuffer();
		const out_state = createStateBuffer();
		const out_logits = createLogitsBuffer();

		// Lets do a single pass
		// 12092 : "Hello"
		let ret = cpp_bind.rwkv_eval(ctx, 12092, in_state, out_state, out_logits);

		// Check if its OK?
		assert.ok(ret);
	});
});
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

const RWKV = require("../src/RWKV")

// Try from the downloaded home dir first, then fallback to ./raven
const fs = require("fs")
const os = require("os");
const path = require("path");
let modelPath = path.join(os.homedir(), '.rwkv', 'raven_1b5_v12_Q8_0.bin');
if( fs.existsSync(modelPath) == false ) {
	modelPath = "./raven/raven_1b5_v12_Q8_0.bin";
}

//-----------------------------------------------------------
// Test data
//-----------------------------------------------------------

// The demo prompt for RWKV
const dragonPrompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

//-----------------------------------------------------------
// And perform the unit tests
//-----------------------------------------------------------

// Validate the R1 action decider step
describe("RWKV.js instance test", function() {
	// Set large timeout
	this.timeout(60 * 1000); // 60 seconds

	// Instance of RWKV to test
	let raven = null;

	// Lets load the model, and get a ctx pointer
	it("Instance setup + model load", async function() {
		// Lets load the model, tagged to the number of threads
		raven = new RWKV(modelPath)
		await raven.setup();

		// Validate the context is not null
		assert.ok(raven != null, "Instance setup completed");
	});

	// Lets do a prompt precomputation
	it("Prompt precomputation", async function() {
		// Lets do a prompt precomputation
		let res = await raven.completion( { 
			prompt: dragonPrompt,
			max_tokens: 0
		} );

		// Assert not null
		assert.ok( res );

		// Log the result
		console.log( "Prompt precomputation - prompt tokens per second : ", res.perf.promptPerSecond );
	});

	// Lets do a prompt and completion
	it("Prompt and completion", async function() {
		// Lets do a prompt precomputation
		let res = await raven.completion( { 
			prompt: dragonPrompt,
			max_tokens: 64
		} );

		// Assert not null
		assert.ok( res );

		// Log the result
		console.log( "Prompt and completion - dragon prompt result : ", res.completion.trim() );
		console.log( "Prompt and completion - prompt tokens per second : ", res.perf.promptPerSecond );
		console.log( "Prompt and completion - completion tokens per second : ", res.perf.completionPerSecond );
	});

});
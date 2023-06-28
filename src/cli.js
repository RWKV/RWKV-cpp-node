#!/usr/bin/env node

/**
 * RWKV-cpp-node CLI example
 * 
 * The following CLI tooling was meant more of a simple fully working demo for RWKV model, as such it opts for simplicity over optimal setups.
 * 
 * Mostly in the following two ways
 * - we use file sync operations for config file, if needed so we do not need to deal with async race conditions
 * - we do a simple, DIY argument parser, to limit the amount of dependencies
 * 
 * If you wish to add even more features to this CLI tool, please consider creating a new project, with this as a dependency instead.
 **/

// ---------------------------
// Check if node version is >= 18
// ---------------------------

const nodeVersion = process.versions.node;
const nodeMajorVersion = Number(nodeVersion.split('.')[0]);
if (nodeMajorVersion < 18) {
	console.error(`Node version ${nodeVersion} is not supported. Please use node >= 18`);
	process.exit(1);
}

// ---------------------------
// Dependencies
// ---------------------------

const os = require('os');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const ProgressBar = require('progress');
const RWKV = require("./RWKV");
const inquirerPromise = import('inquirer');

// ---------------------------
// Configs and paths
// ---------------------------

/**
* Prequantized raven .cpp models and their paths
*/
const RWKV_MODELS = require("./rwkv_models");
const RWKV_CLI_DIR = path.join(os.homedir(), '.rwkv');
const CONFIG_FILE = path.join(RWKV_CLI_DIR, 'config.json');

// ---------------------------
// Config file handling
// ---------------------------

// local config object
let _configObj = null;

/**
 * Load the config object from the config file
 * if it is not yet loaded
 * @returns {Object} config object
 */
function loadConfigObject() {
	// Return cached config object
	if( _configObj != null ) {
		return _configObj;
	}

	// Fallback to blank, if no config file found
	if (!fs.existsSync(CONFIG_FILE)) {
		_configObj = {};
		return _configObj;
	}

	// Load config file
	const content = fs.readFileSync(CONFIG_FILE, { encoding: 'utf-8' });
	if (!content) {
		return null;
	}
	_configObj = Object.assign({}, JSON.parse(content));
	return _configObj;
}

/**
 * Save the config object to the config file
 * Merges the values with the existing config file
 * @param {Object} inConfigObj
 **/
function saveConfigObject(inConfigObj) {
	if (!fs.existsSync(RWKV_CLI_DIR)) {
		fs.mkdirSync(RWKV_CLI_DIR);
	}
	_configObj = Object.assign({}, loadConfigObject(), inConfigObj);
	fs.writeFileSync(CONFIG_FILE, JSON.stringify(_configObj), {
		encoding: 'utf-8',
	});
	return _configObj;
}

// ---------------------------
// Model downloading
// ---------------------------

function saveConfigModelName(modelName) {
	saveConfigObject({ model: modelName });
}
function loadConfigModelName() {
	return loadConfigObject().model;
}
function getModelPath() {
	return path.resolve(RWKV_CLI_DIR, loadConfigModelName());
}

/**
* Prompt the user to select a model to download
**/
async function promptModelSelection() {
	// [32m = green
	console.log("\x1b[32m--------------------------------------")
	console.log('RWKV Raven models will be downloaded into ~/.rwkv/');
	console.log('Listed file sizes + 2 : is the approximate amount of RAM your system will need');
	console.log("--------------------------------------\x1b[0m")
	const choices = RWKV_MODELS.map((model) => ({
		name: `${(model.size/1024/1024/1024).toFixed(2)} GB - ${model.label}`,
		value: model,
	}));
	const { model } = await (await inquirerPromise).default.prompt({
		type: 'list',
		name: 'model',
		message: 'Select a RWKV raven model to download: ',
		choices,
	});
	return model;
}

/**
* Given the model config object, download it. Does not check if there is an existing file
* @param {Object} model 
* @returns 
*/
async function downloadModelRaw(model) {
	const destinationPath = path.join(RWKV_CLI_DIR, `${model.name}`);
	console.log(`Downloading '${model.label}' - this will be saved to ${destinationPath}`);
	
	const response = await fetch(model.url);
	if (!response.ok) {
		throw new Error(`Failed to download ${model.label} model: ${response.statusText}`);
	}
	const fileSize = Number(response.headers.get('content-length'));

	// Incremental downloaded size and speed
	let downloadedSize = 0;
	let downloadedSize_gb = 0;
	let speed = 0;

	const progressBar = new ProgressBar('[:bar] :percent :etas - :downloadedSize_gb GB - :speed MB/s', {
		complete: '=',
		incomplete: '-',
		width: 20,
		total: fileSize,
	});
	const outputStream = fs.createWriteStream(destinationPath);
	const reader = response.body.getReader();

	async function processData() {
		const startTime = Date.now();
		while (true) {
			const { done, value } = await reader.read();

			// Check for completion, if so terminate the loop
			if (done) {
				progressBar.terminate();
				console.log(`Model ${model.name} downloaded to ${destinationPath}`);
				outputStream.end();
				reader.releaseLock();
				break;
			}

			// Calculate size and speed
			downloadedSize += value.length;
			downloadedSize_gb = (downloadedSize / 1024 / 1024 / 1024).toFixed(2);
			const timeTaken = (Date.now() - startTime) / 1000;
			speed = (downloadedSize / 1024 / 1024 / timeTaken).toFixed(2);

			progressBar.tick(value.length, {
				speed: speed,
				downloadedSize_gb: downloadedSize_gb,
			});
				
			// Await for outputStream.write to complete
			await new Promise((resolve, reject) => {
				outputStream.write(value, (err) => {
					if (err) {
						reject(err);
						return;
					}
					resolve();
				});
			});
		}
	}
	await processData();

	// Validate the model
	console.log(`Validating downloaded model ...`)
	if( (await validateModel(model)) == false ) {
		console.log(`Model validation failed, run the --setup command again (mismatched size/sha256)`)
		process.exit(1);
	} else {
		console.log(`Model validation passed!`)
	}
	return destinationPath;
}

/**
* Given a file path, compute the sha256 hash of the file
* @param {String} filePath 
* @returns 
*/
function sha256File(filePath) {
	const hash = crypto.createHash('sha256');
	const input = fs.createReadStream(filePath);
	
	return new Promise((resolve, reject) => {
		input.on('readable', () => {
			const data = input.read();
			if (data) {
				hash.update(data);
			} else {
				resolve(hash.digest('hex'));
			}
		});
		
		input.on('error', reject);
	});
}

/**
 * Validate if the downloaded model matches, and fail if it doesn't
 * @param {*} model 
 */
async function validateModel(model) {
	const modelPath = path.join(RWKV_CLI_DIR, `${model.name}`);
	
	// If the model already exists, return it
	if (fs.existsSync(modelPath)) {
		const stats = fs.statSync(modelPath);
		if (stats.size === model.size) {
			// Check if the file is the right sha256 hash
			const hash = await sha256File(modelPath);
			if (hash === model.sha256) {
				return true;
			}
		}
	}
	return false;
}

/**
* Given the model config object, download it
* @param {Object} model 
* @returns 
*/
async function downloadIfNotExists(model) {
	// The model download directory
	if (!fs.existsSync(RWKV_CLI_DIR)) {
		fs.mkdirSync(RWKV_CLI_DIR);
	}
	const modelPath = path.join(RWKV_CLI_DIR, `${model.name}`);
	
	// If the model already exists, return it
	if (fs.existsSync(modelPath)) {
		const stats = fs.statSync(modelPath);
		if (stats.size === model.size) {
			console.log(`Model ${model.name} already exists at ${modelPath} : checking sha256 hash ...`);
			
			// Check if the file is the right sha256 hash
			const hash = await sha256File(modelPath);
			if (hash === model.sha256) {
				console.log(`Model ${model.name} already exists at ${modelPath} : hash matches, skipping download`);
				return modelPath;
			}
			
			// Wrong hash handling
			console.log(`Model ${model.name} exists but has the wrong hash. Re-downloading ...`);
			fs.unlinkSync(modelPath);
		} else {
			// Check if the file is NOTthe correct size
			console.log(`Model ${model.name} exists but is the wrong size. Re-downloading ...`);
			fs.unlinkSync(modelPath);
		}
	}
	
	// Download the model
	return await downloadModelRaw(model);
}

// ---------------------------
// Other config setup
// ---------------------------

/**
* Prompt the user to select a model to download
**/
async function promptThreadCount() {
	// Get the vCPU count
	const vCPUCount = os.cpus().length;

	// [32m = green
	console.log("\x1b[32m--------------------------------------")
	console.log('For most systems, the optimal number of threads = 4');
	console.log('This is due to RAM speed limitations, where more threads may even hurt performance.');
	console.log('(try different settings, and see what works best for you)');
	console.log("--------------------------------------\x1b[0m")

	const { threads } = await (await inquirerPromise).default.prompt({
		type: 'number',
		name: 'threads',
		message: 'How many CPU threads should be used to run the model?',
		default: Math.min(loadConfigObject().threads || 4, vCPUCount),
	});

	// Save the value
	saveConfigObject({ threads: Math.max(threads,1) });
	return threads;
}

/**
* Prompt the user to select a model to download
**/
async function promptGpuOffload() {
	// Get current model path
	const modelPath = getModelPath();

	// Get the model file size
	const modelSize = fs.statSync(modelPath).size;
	const modelSize_gb = (modelSize/ 1024 / 1024 / 1024).toFixed(2)

	// [32m = green
	console.log("\x1b[32m--------------------------------------")
	console.log('[Experimental] Offload a part of the model computation to the GPU');
	console.log("This can be configured either in percentage (eg. 50%), or number of layers (eg. 12)");
	console.log("");
	console.log("You will approximately need the % configured, of your model disk size + 4GB of VRAM to run the model");
	console.log("(For example: a 50% of a 24GB model, will require 12GB + 4GB of VRAM to run)");
	console.log("");
	console.log(`You current model size is ${modelSize_gb} GB in size, and will require the following estimated vRAM for ...`);
	console.log("   25% offload: " + (modelSize_gb * 0.25 + 4).toFixed(2) + " GB");
	console.log("   50% offload: " + (modelSize_gb * 0.5 + 4).toFixed(2) + " GB");
	console.log("  100% offload: " + (modelSize_gb * 1.0 + 4).toFixed(2) + " GB");
	console.log("--------------------------------------\x1b[0m")

	const { gpu } = await (await inquirerPromise).default.prompt({
		type: 'number',
		name: 'gpu',
		message: 'How many % of the model layers to do gpu offloading?',
		default: parseInt(loadConfigObject().gpuOffload) || 0,
	});
	let gpuOffload =  Math.max(gpu,0)+"%"

	// Save the value
	saveConfigObject({ gpuOffload:gpuOffload });
	return gpuOffload;
}

/**
 * Perform the full setup sequence
 */
async function performSetup() {
	// First ask for the model to download
	const model = await promptModelSelection();

	// Download the model (if needed)
	await downloadIfNotExists(model);
	await saveConfigModelName(model.name);

	// Configure the remaining settings
	let threads = await promptThreadCount();
	let gpu = await promptGpuOffload();

	console.log("\x1b[32m--------------------------------------")
	console.log(`Final configured settings:`)
	console.log(`  Model Path:  ${getModelPath()}`);
	console.log(`  Threads:     ${threads}`);
	console.log(`  GPU Offload: ${gpu}`);
	console.log("--------------------------------------\x1b[0m")
}

// ---------------------------
// Model Execution
// ---------------------------

async function startChatBot(rwkvConfig) {
	
	// Load the chatbot
	console.log("\x1b[32m--------------------------------------")
	console.log(`Starting RWKV chat mode`)
	console.log("--------------------------------------")
	console.log(`Loading model from ${rwkvConfig.path} ...`)

	const raven = new RWKV(rwkvConfig);
	await raven.setup();

	// User / bot label name
	const user = "User";
	const bot = "Bot";
	const interface = ":";

	// The chat bot prompt to use
	const prompt = [
		"",
		`The following is a verbose detailed conversation between ${user} and a young women ${bot}. ${bot} is intelligent, friendly and cute. ${bot} is unlikely to disagree with ${user}.`,
		"",
		`${user}${interface} Hello ${bot}, how are you doing?`,
		"",
		`${bot}${interface} Hi ${user}! Thanks, I'm fine. What about you?`,
		"",
		`${user}${interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?`,
		"",
		`${bot}${interface} Not at all! I'm listening.`,
		"",
		""
	].join("\n");

	// Preload the prompt, this helps make the first response faster
	let firstLoadPromise = raven.preloadPrompt(prompt);

	// Log the start of the conversation
	console.log(`The following is a conversation between the ${user} and the ${bot} ...`)
	console.log("--------------------------------------\x1b[0m")

	// The chat history
	let chatHistory = prompt;

	// Lets start the loop
	while(true) {
		// Get the user input
		let res = await (await inquirerPromise).default.prompt([{
			type: 'input',
			name: 'userInput',
			message: `${user}${interface} `,
			validate: (value) => {
				return (value||"").trim().length > 0;
			}
		}]);

		// Ensure first load finished
		await firstLoadPromise;

		// Add the user input to the chat history
		chatHistory += `${user}${interface} ${res.userInput}\n\n${bot}:`;

		// Run the completion
		process.stdout.write(`${bot}: `);
		res = await raven.completion({
			prompt: chatHistory,
			max_tokens: 2000,
			streamCallback: (text) => {
				process.stdout.write(text);
			},
			stop: [`\n${bot}:`, `\n${user}:`]
		});
		// console.log(res);
		chatHistory += `${res.completion.trim()}\n\n`;
	}
}

async function runDragonPrompt(rwkvConfig, size=100) {
	// Load the chatbot
	console.log(`\x1b[32mLoading model with ${JSON.stringify(rwkvConfig)} ...`)
	console.log("--------------------------------------\x1b[0m")
	const raven = new RWKV(rwkvConfig);
	await raven.setup();

	// The demo prompt for RWKV
	const dragonPrompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'
	console.log(`Running the test dragon prompt: ${dragonPrompt}`);

	// Does the actual demo execution
	let res = await raven.completion({
		prompt: dragonPrompt,
		max_tokens: size,
		streamCallback: (text) => {
			process.stdout.write(text);
		}
	});

	// Log the stats
	console.log("");
	console.log("\x1b[32m--------------------------------------\x1b[0m")
	console.log(`\n\nRWKV perf stats: \n${JSON.stringify(res.perf)}`);
}

// ---------------------------
// CLI handling
// ---------------------------

// Run the CLI within an async function
(async function() {
	// Get the current CLI arguments
	const args = process.argv.slice(2);
	
	// Perform the setup operation
	if (args[0] === '--setup') {
		console.log("--setup call detected, starting setup process...");
		await performSetup();
		return;
	}

	// The model path to use
	let modelPath = null;

	// Check if first arg is --modelPath
	if (args.indexOf('--modelPath') >=0) {
		modelPath = args[1];
	}

	// If model path is not set, check if it is in the config
	if(modelPath == null) {
		// Get the current config
		let modelName = loadConfigModelName();
		
		// Check if first arg is --setup
		if (modelName == null) {
			console.log("Missing model name, performing first time setup (this will be skipped for subsequent calls)...");
			await performSetup();
			modelName = loadConfigModelName();
		}
	
		// Get the model config
		const model = RWKV_MODELS.find((model) => model.name === modelName);
		if (!model) {
			console.error(`Model ${modelName} not found, you may need to run --setup again`);
			return;
		}

		// Set the model path
		modelPath = path.resolve(RWKV_CLI_DIR, model.name);
	}

	// check if args contains '--threads' and set threads to the next arg if not specified default to 6
	let threads = loadConfigObject().threads || 4;
	if (args.indexOf('--threads') >= 0) {
		let _threads = args[args.indexOf('--threads') + 1];
		try {
			threads = parseInt(_threads);
			if (threads > 0) {
				threads = _threads;
			} else {			
				throw new Error("Invalid threads value, should be positive, defaulting to 6");
				threads = 6;
			}
		} catch (error) {
			console.log("Invalid threads value, defaulting to 6");
			threads = 6;
		}
	}
	
	// Check if '--gpu' is specified and set vram to the next arg if not specified default to 0
	let gpuOffload = loadConfigObject().gpuOffload || 0;
	if (args.indexOf('--gpu') >= 0) {
		let _gpuOffload = args[args.indexOf('--gpu') + 1];
		try {
			let _parsedGpuOffload = parseInt(_gpuOffload);
			if (isNaN(_parsedGpuOffload) || !isFinite(_parsedGpuOffload)) {		
				console.warn("Invalid gpu offload value, " + _gpuOffload +  " defaulting to "+gpuOffload);
			}
			gpuOffload = _gpuOffload;
		} catch (error) {
			console.warn("Invalid layers value, " + _gpuOffload + " defaulting to "+gpuOffload);	
		}
	}

	// RWKV config
	let rwkvConfig = {
		path: modelPath,
		threads: threads,
		gpuOffload: gpuOffload
	};

	// Check if the --dragon test prompt is running
	if(args[0] === '--dragon' || args[1] === '--dragon' || args[2] === '--dragon') {
		let size = 1000;
		if (args.indexOf('--size') >= 0) {
			let _size = args[args.indexOf('--size') + 1];
			try {
				size = parseInt(_size);
				if (size > 0) {
					size = _size;
				} else {
					console.warn("Invalid test size value, should be positive, defaulting to 1000");
					size = 100;
				}
			} catch (error) {
				console.warn("Invalid test size value, defaulting to 1000");
				size = 100;
			}
		}
		await runDragonPrompt(rwkvConfig, size);
		return;
	}

	// Call the main start chat bot instead
	await startChatBot(rwkvConfig);
})()

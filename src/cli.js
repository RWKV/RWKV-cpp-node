#!/usr/bin/env node

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
const DOWNLOAD_CHUNK_SIZE = 1024;

let threadCount = 6;
let layers = 0;

// ---------------------------
// Model downloading
// ---------------------------

/**
* Prompt the user to select a model to download
**/
async function promptModelSelection() {
	console.log(`--------------------------------------`)
	console.log('RWKV Raven models will be downloaded into ~/.rwkv/');
	console.log('Listed file sizes + 2 : is the approximate amount of RAM your system will need');
	console.log(`--------------------------------------`)
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

function saveConfigModelName(modelName) {
	if (!fs.existsSync(RWKV_CLI_DIR)) {
		fs.mkdirSync(RWKV_CLI_DIR);
	}
	const config = {
		model: modelName,
	};
	fs.writeFileSync(CONFIG_FILE, JSON.stringify(config), {
		encoding: 'utf-8',
	});
}

function loadConfigModelName() {
	if (!fs.existsSync(CONFIG_FILE)) {
		return null;
	}
	const content = fs.readFileSync(CONFIG_FILE, { encoding: 'utf-8' });
	if (!content) {
		return null;
	}
	return JSON.parse(content)?.model;
}

/**
 * Perform the full setup sequence
 */
async function performSetup() {
	// First ask for the model to download
	const model = await promptModelSelection();

	// Download the model (if needed)
	await downloadIfNotExists(model);

	// Save the config
	saveConfigModelName(model.name);
}

// ---------------------------
// Model Execution
// ---------------------------

async function startChatBot(modelPath) {
	


	// Load the chatbot
	console.log(`--------------------------------------`)
	console.log(`Starting RWKV chat mode`)
	console.log(`--------------------------------------`)
	console.log(`Loading model from ${modelPath} ...`)

	const raven = new RWKV(modelPath, threadCount, layers);

	// User / bot label name
	const user = "User";
	const bot = "Bot";
	const interface = ":";

	// The chat bot prompt to use
	const prompt = [
		
			"",
		`\nThe following is a verbose and detailed conversation between an AI assistant called Bot, and a human user called User.`,
		"",
		`Bot is intelligent, knowledgeable, wise and polite.`,
		"",
		`\n\nUser: french revolution what year\n\n`,
		"",
		`Bot: The French Revolution started in 1789, and lasted 10 years until 1799.\n\n`,
		"",
		`User: 3+5=?\n\n`,
		"",
		`Bot: The answer is 8.\n\n`,
		"",
		`User: guess i marry who ?\n\n`,
		"",
		`Bot: Only if you tell me more about yourself - what are your interests?\n\n`,
		"",
		`User: solve for a: 9-a=2\n\n`,
		"",
		`Bot: The answer is a = 7, because 9 - 7 = 2.\n\n`,
		"",
		`User: wat is lhc\n\n`,
		"",
		`Bot: LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.\n\n"`
		
	/*
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
	*/
	].join("\n");

	// Preload the prompt
	console.log(`Preloading the prompt: ${prompt}`);
	raven.preloadPrompt(prompt);

	// Log the start of the conversation
	console.log(`The following is a conversation between ${user} the user and ${bot} the chatbot.`)
	console.log(`--------------------------------------`)

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

		// Add the user input to the chat history
		chatHistory += `${user}${interface} ${res.userInput}\n\n${bot}:`;

		// Run the completion
		process.stdout.write(`${bot}: `);
		res = raven.completion({
			prompt: chatHistory,
			max_tokens: 200,
			streamCallback: (text) => {
				process.stdout.write(text);
			},
			stop: ["\nuser:", "\nUser:"]
		});
		// console.log(res);
		chatHistory += `${res.completion.trim()}\n\n`;
	}
}

async function runDragonPrompt(modelPath) {
	// Load the chatbot
	console.log(`Loading model from ${modelPath} ...`)
	console.log(`--------------------------------------`)
	const raven = new RWKV(modelPath, threadCount, layers);

	// The demo prompt for RWKV
	const dragonPrompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'
	console.log(`Running the test dragon prompt: ${dragonPrompt}`);
	raven.preloadPrompt(dragonPrompt);

	// Does the actual demo execution
	let res = raven.completion({
		prompt: dragonPrompt,
		max_tokens: 1000,
		streamCallback: (text) => {
			process.stdout.write(text);
		}
	});

	// Log the stats
	console.log("");
	console.log(`--------------------------------------`)
	console.log(`\n\nRWKV perf stats: \n${JSON.stringify(res.perf)}`);
}

// ---------------------------
// CLI handling
// ---------------------------

class rwkv_model {
	constructor(modelpath, threadcount, layers) {
	  this.modelpath = modelpath;
	  this.threadcount = threadcount;
	  this.layers = layers;
	}
  }

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

	// check if args contains '--threadCount' and set threadcount to the next arg if not specified default to 6

	if (args.indexOf('--threadcount') >= 0) {
		let _threadCount = args[args.indexOf('--threadcount') + 1];
		try {
			threadCount = parseInt(_threadCount);
			if (threadCount > 0) {
				threadCount = _threadCount;
			} else {			
				throw new Error("Invalid threadCount value, should be positive, defaulting to 6");
				threadCount = 6;
			}
		} catch (error) {
			console.log("Invalid threadCount value, defaulting to 6");
			threadCount = 6;
		}
	}
	
	// Check if '--layers' is specified and set vram to the next arg if not specified default to 0
	if (args.indexOf('--layers') >= 0) {
		layers = 0;

		_layers = args[args.indexOf('--layers') + 1];
		try {
			layers = parseInt(_layers);
		
			if (isNaN(layers) || !isFinite(layers)) {		
				throw new Error("Invalid layers value, " + layers +  " defaulting to 0");
			}		
		} catch (error) {
			console.log("Invalid layers value, " + _layers + " defaulting to 0");	
			layers = 0;
		}
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

	// Check if the --dragon test prompt is running
	if(args[0] === '--dragon' || args[1] === '--dragon' || args[2] === '--dragon') {
		await runDragonPrompt(modelPath);
		return;
	}
	modelArgs = new rwkv_model(modelPath, parseInt(threadCount), layers);
//todo: find why i need to parse int here again.

	// Call the main start chat bot instead
	await startChatBot(modelPath);
})()

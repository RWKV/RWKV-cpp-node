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
const RAVEN_MODELS = [
	{
		label: 'RWKV raven 1B5 v11 (Small, Fast)',
		name: "raven_1b5_v11.bin",
		url: "https://huggingface.co/datasets/picocreator/rwkv-4-cpp-quantize-bin/resolve/main/RWKV-4-Raven-1B5-v11.bin",
		sha256: "098c6ea8368f68317c99283195651685bdaac417857a21e447eadced2e62f8eb",
		size: 3031328341
	},
	{ 
		label: 'RWKV raven 7B v11 (Q8_0)', 
		name: "raven_7b_v11_Q8_0.bin",
		url: 'https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/Q8_0-RWKV-4-Raven-7B-v11x-Eng99%25-Other1%25-20230429-ctx8192.bin',
		sha256: '75d252da63405e9897bff2957f9b6b1c94d496a50e4772d5fc1ec22fb048f9b5',
		size: 8681332157
	},
	{ 
		label: 'RWKV raven 7B v11 (Q8_0, multilingual, performs slightly worse for english)', 
		name: "raven_7b_v11_Q8_0_multilingual.bin",
		url: 'https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/Q8_0-RWKV-4-Raven-7B-v11-Eng49%25-Chn49%25-Jpn1%25-Other1%25-20230430-ctx8192.bin',
		sha256: 'ee4a6e7fbf9c2bd3558e4a92dbf16fd25d8599c7bef379ba91054981a8f665e0',
		size: 8681332157
	},
	{ 
		label: 'RWKV raven 14B v11 (Q8_0)',
		name: "raven_14b_v11_Q8_0.bin",
		url: 'https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/Q8_0-RWKV-4-Raven-14B-v11x-Eng99%25-Other1%25-20230501-ctx8192.bin',
		sha256: '1cb56cd7784264a8f0ed2efd17a97f4866075915f2b1a8497c1abb826edade8b',
		size: 16374220069
	},
	{ 
		label: 'RWKV Pile 169M (Q8_0, lacks instruct tuning, use only for testing)',
		name: "rwkv_169M_pileplus_Q8_0.bin",
		url: 'https://huggingface.co/datasets/picocreator/rwkv-4-cpp-quantize-bin/resolve/main/RWKV-4-PilePlus-169M-Q8_0.bin',
		sha256: '82c2949f6f9261543b13cbd1409fd2069cd67d9e2ad031bb727bb0bd43527af1',
		size: 258391865
	}
];

const RWKV_CLI_DIR = path.join(os.homedir(), '.rwkv');
const CONFIG_FILE = path.join(RWKV_CLI_DIR, 'config.json');
const DOWNLOAD_CHUNK_SIZE = 1024;

// ---------------------------
// Model downloading
// ---------------------------

/**
* Prompt the user to select a model to download
**/
async function promptModelSelection() {
	console.log('RWKV model will be downloaded into ~/.rwkv/');
	const choices = RAVEN_MODELS.map((model) => ({
		name: `${model.label} - ${(model.size/1024/1024/1024).toFixed(2)} GB`,
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
			outputStream.write(value);
		}
	}
	await processData();

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

	const raven = new RWKV(modelPath);

	// User / bot label name
	const user = "Bob";
	const bot = "Alice";
	const interface = ":";

	// The chat bot prompt to use
	const prompt = [
		"",
		`The following is a verbose detailed conversation between ${user} and a young girl ${bot}. ${bot} is intelligent, friendly and cute. ${bot} is unlikely to disagree with ${user}.`,
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

	// Preload the prompt
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
			stop: ["\nBob:", "\nbob:"]
		});
		// console.log(res);
		chatHistory += `${res.completion.trim()}\n\n`;
	}
}

async function runDragonPrompt(modelPath) {
	// Load the chatbot
	console.log(`Loading model from ${modelPath} ...`)
	console.log(`--------------------------------------`)
	const raven = new RWKV(modelPath);

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
	console.log(`--------------------------------------`)
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
	if (args[0] === '--modelPath') {
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
		const model = RAVEN_MODELS.find((model) => model.name === modelName);
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

	// Call the main start chat bot instead
	await startChatBot(modelPath);
})()

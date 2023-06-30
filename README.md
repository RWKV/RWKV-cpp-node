# RWKV.cpp NodeJS bindings

Arguably the easiest way to get RWKV.cpp running on node.js

This project primary use case, is to be used as a nodejs library for running RWKV.cpp
The CLI tooling, is simply a helper tooling, to do quick demo's or benchmark via node.js locally

**World model is not yet supported**

## Running it as a JS lib
```.js
const RWKV = require("RWKV-cpp-node");

// Load the module with the pre-qunatized cpp weights
const raven = new RWKV("<path-to-your-model-bin-files>")

// You must call the setup before completion
await raven.setup();

// Call the completion API
let res = await raven.completion("RWKV is a")

// And log, or do something with the result
console.log( res.completion )
```

## Running it as a CLI
```.bash
# Install globally
npm install -g rwkv-cpp-node

# This will start the interactive CLI, 
# which will guide you in downloading, and running the chat model
rwkv-cpp-node --setup

# You can run the chat model, with thread count / gpu offload %
rwkv-cpp-node --threads 4 --gpu 100%

# For benchmarking
rwkv-cpp-node --threads 4 --gpu 0 --dragon --size 100
```

> This is not a pure JS solution, and depends on the [precompiled RWKV.cpp binaries found here](https://github.com/saharNooby/rwkv.cpp)
>
> This currently runs purely on your CPU, while that means you can use nearly anything to run it, you also do not get any insane speed up with a GPU (yet)
>
> Additionally V2 breaks compatiblity with V1, due to changes in quantization weights.

# What is RWKV?

RWKV, is a LLM which which can switch between "transformer" and "RNN" mode.

This gives it the best of both worlds
- High scalable training in transformer
- Low overheads when infering each token in RNN mode

Along with the following benefits
- Theoretically Infinite context size
- Embedding support via hidden states

For more details on the math involved, and how this model works on a more technical basis. [Refer to the official project](https://github.com/BlinkDL/RWKV-LM)

# JS CLI demo

If you just want to give it a spin, the fastest way is to use npm.
First perform the setup (it will download the RWKV files into your home directory)

```.bash
# Install globally
npm install -g rwkv-cpp-node

# First run the setup
rwkv-cpp-node --setup
```

You can then choose a model to download ...

```
--setup call detected, starting setup process...
RWKV model will be downloaded into ~/.rwkv/
? Select a RWKV raven model to download:  (Use arrow keys)
❯ RWKV raven 1B5 v11 (Small, Fast) - 2.82 GB 
  RWKV raven 7B v11 (Q8_0) - 8.09 GB 
  RWKV raven 7B v11 (Q8_0, multilingual, performs slightly worse for english) - 8.09 GB 
  RWKV raven 14B v11 (Q8_0) - 15.25 GB 
  RWKV Pile 169M (Q8_0, lacks instruct tuning, use only for testing) - 0.24 GB 
```

> PS: The file size equals to the approximate amount of storage and ram your system needs

Subsequently, you can run the interactive chat mode

```.bash
# Load the interactive chat
rwkv-cpp-node
```

Which would start an interactive shell session, with something like the following

```
--------------------------------------
Starting RWKV chat mode
--------------------------------------
Loading model from /root/.rwkv/raven_1b5_v11.bin ...
The following is a conversation between the user and bot
--------------------------------------
? User: Hi
Bot:    How can I help you?
? User: Tell me something interesting about ravens
Bot:    RAVEN. I am most fascinated by the raven because of its incredible rate of survival. Ravens have been observed to live longer than any other bird, rumored to reach over 200 years old. They have the ability to live for over 1,000 years, a remarkable feat. This makes them the odd man out among birds!
```

> PS: RWKV like all chat models, can and do lie about stuff.

Finally if you want to run a custom model, or just run the benchmark

```.bash
# If you want to run with a pre downloaded model
rwkv-cpp-node --modelPath "<path to the model bin file>"

# If you want to run the "--dragon" prompt benchmark
rwkv-cpp-node --dragon
rwkv-cpp-node --modelPath "<path to the model bin file>" --dragon
```

# JS Lib Setup 

Install the node module

```.bash
npm i rwkv-cpp-node
```

Download one of the prequantized rwkv.cpp weights, from hugging face (raven, is RWKV pretrained weights with fine-tuned instruction sets)
- [RWKV-4-rave-ggml-quantized](https://huggingface.co/latestissue/rwkv-4-raven-ggml-quantized/)

Alternatively you can download one of the [raven pretrained weights from the hugging face repo](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main). 
And perform your own quantization conversion using the [original rwkv.cpp project](https://github.com/saharNooby/rwkv.cpp)

# JS Usage Details

The JS interface for the RWKV model is async/promises based

```.javascript
const RWKV = require("RWKV-cpp-node");

// Load the module with the pre-qunatized cpp weights
const raven = new RWKV("<path-to-your-model-bin-files>")

// You must call the setup before completion
await raven.setup();

// Call the completion API
let res = await raven.completion("RWKV is a")

// And log, or do something with the result
console.log( res.completion )
```

Advance setup options

```.javascript
// You can setup with the following parameters with a config object (instead of a string path)
const raven = new RWKV({
	// Path to your cpp weights
	path: "<path-to-your-model-bin-files>",

	// Threads count to use, this is auto detected based on your number of vCPU
	// if its not configured, uses 4 with gpu offloading, else uses half of vCPU detected
	threads: 4,

	// Number of layers (eg. 12), or % of the model (eg: 50%) to offload to the gpu
	// defaults: 0
	gpuOffload: 0,

	// Number of concurrent inferences, the model is cloned while sharing the weights
	// for each concurrent instances configured. This is only useful in server prod env
	// deafults: 1
	concurrent: 1,

	// Batch size of the input to process, this is only useful with gpuOffload
	// Defaults to 64 with gpuOffload, else 1
	// ---
	// batchSize: 64,

	//
	// Cache size for the RKWV state, This help optimize the repeated RWKV calls
	// in use cases such as "conversation", allow it to skip the previous chat computation
	//
	// it is worth noting that the 7B model takes up about 2.64 MB for the state buffer, 
	// meaning you will need atleast 264 MB of RAM for a cachesize of 100
	//
	// This defaults to 50
	// Set to false or 0 to disable
	//
	stateCacheSize: 50
});
await raven.setup();
```

Completion API options

```.javascript
// Lets perform a completion, with more options
let res = await raven.completion({

	// The prompt to use
	prompt: "<prompt str>",

	// Completion default settings
	// See openai docs for more details on what these do for your output if you do not understand them
	// https://platform.openai.com/docs/api-reference/completions
	max_tokens: 64,
	temperature: 1.0,
	top_p: 1.0,
	stop: [ "\n" ],

	// Streaming of output, either token by token, or the full complete output stream
	streamCallback: function(tokenStr, fullCompletionStr) {
		// ....
	},

	// Existing RWKV hidden state, represented as a Flaot32Array
	// do not use this unless you REALLY KNOW WHAT YOUR DOING
	//
	// This will skip the state caching logic 
	initState: (Special Float32Array)
});

// Additionally if you have a commonly reused instruction set prefix, you can preload this
// using either of the following (requires the stateCacheSize to not be disabled)
await raven.preloadPrompt( "<prompt prefix string>" )
await raven.completion({ prompt:"<prompt prefix string>", max_tokens:0 })
```

Completion output format

```.javascript
// The following is a sample of the result object format
let resFormat = {
	// Completion generated
	completion: '<completion string used>',

	// Prompt used
	prompt: '<prompt string used>',

	// Token usage numbers
	usage: {
		promptTokens: 41,
		completionTokens: 64,
		totalTokens: 105,
		// number of tokens in the prompt that was previously cached
		promptTokensCached: 39 
	},

	// Performance statistics of the completion operation
	//
	// the following perf numbers is from a single 
	// `Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz`
	// an old 2014 processor, with 28 vCPU 
	// with the 14B model Q8_0 quantized
	// 
	perf: {
		// Time taken in ms for each segment
		promptTime: 954,
		completionTime: 35907,
		totalTime: 36861,

		// Time taken in ms to process each token at the respective phase
		timePerPrompt: 477, // This excludes cached tokens
		timePerCompletion: 561.046875,
		timePerFullPrompt: 23.26829268292683, // This includes cached tokens (if any)

		// The average tokens per second
		promptPerSecond: 2.0964360587002098, // This excludes cached tokens
		completionPerSecond: 1.7823822652964603,
		fullPromptPerSecond: 42.9769392033543 // This includes cached tokens (if any)
	}
}
```

# Want lower level CPP based binding access?

You can call our cpp_bind interface code via

```
const cpp_bind = require("rwkv-cpp-node").cpp_bind;

// You can find the code here : https://github.com/RWKV/RWKV-cpp-node/blob/main/src/cpp_bind.js
```

# What can be improved?

- ~~[Add GPU support via RWKV-cpp-cuda project](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda)~~
- [RWKV-tokenizer-node library performance](https://github.com/PicoCreator/RWKV-tokenizer-node/issues/1)
- ~~[Add MMAP support for RWKV.cpp](https://github.com/saharNooby/rwkv.cpp/issues/50)~~
- ~~[Reducing JS and RWKV.cpp back and forth for prompt eval](https://github.com/saharNooby/rwkv.cpp/pull/49)~~
- Validate and add support for X arch / OS 
	- If your system is not supported, try to do a build on the rwkv.cpp project
		- [Add it to the lib folder](https://github.com/PicoCreator/RWKV-cpp-node/tree/main/lib)
		- [Modify the OS / Architecture detection code](https://github.com/PicoCreator/RWKV-cpp-node/blob/main/src/cpp_bind.js#L19)
- ~~Utility function to download the model weights / quantize them ??~~
- ~~CLI tooling to quickly setup / download ??~~
- varient of `preloadPrompt` which gurantee the saved prompt does not get cache evicted ??

# Known issues

- You need macOS 12 and above

# How to run the unit test?

```.bash
# Download the test model
mkdir -p ./raven/
wget -O raven_1b5_v12_Q8_0.bin https://huggingface.co/latestissue/rwkv-4-raven-ggml-quantized/resolve/main/q8_0-RWKV-4-Raven-1B5-v12-Eng98%25-Other2%25-20230520-ctx4096.bin 

# Run the test
npm run test
```

# Designated maintainer

[@picocreator](https://github.com/PicoCreator) - is the current maintainer of the project, ping him on the RWKV discord if you have any questions on this project

# Special thanks & refrences

@saharNooby - original rwkv.cpp implementation

- https://github.com/saharNooby/rwkv.cpp

@BlinkDL - for the main rwkv project

- https://github.com/BlinkDL/RWKV-LM

# [ THIS IS OUTDATED ] Time taken per token completion for RWKV.cpp v1

| Model Size | Download Size | RAM usage | AWS c6g.4xlarge (arm64, 8 Core, 16 vCPU) | AWS c6gd.16xlarge (arm64, 32 Core, 64 vCPU) | M2 Pro, Mac Mini  (6 P core + 4 E core) | Oracle A1 (4 Cores) | AMD Ryzen 7 3700X (x64, 8 Core, 16 vCPU) |
|------------|---------------|-----------|------------------------------------------|---------------------------------------------|-----------------------------------------|---------------------|------------------------------------------|
| 1.5B       | 2.82 GB       | ~ 3.0 GB  | 94.699 ms                                | 81.497 ms                                   | 57.448 ms                               | 177.025 ms          | 283.681 ms                               |
| 3B         | 5.56 GB       | ~ 5.7 GB  | 139.038 ms                               | 109.676 ms                                  | 103.013 ms                              | 317.793 ms          | 564.116 ms                               |
| 7B (Q5_1)  | 5.65 GB       | ~ 7.1 GB  |                                          |                                             | 180.137 ms                              | 482.916 ms          |                                          |
| 7B (Q8_0)  | 8.09 GB       | ~ 8.3 GB  | 167.148 ms                               | 126.856 ms                                  | 140.261 ms                              | 382.687 ms          | 406.984 ms                               |
| 7B         | 13.77 GB      | ~ 14.9 GB | 259.888 ms                               | 175.069 ms                                  | 210.280 ms                              | 733.818 ms          | 729.948 ms                               |
| 14B (Q8_0) | 15.25 GB      | ~ 16.4 GB | 269.201 ms                               | 199.114 ms                                  | 243.889 ms                              | 688.014 ms          | 738.947 ms                               |
| 14B        | 26.36 GB      | ~ 27.9 GB | 460.963 ms                               | 273.277 ms                                  |                                         | 883.386 ms          |                                          |

** Note: There are know performance bottleneck issue in the tokenizer, and sampler written in nodejs, as its a single threaded operation, between each "token" in nodejs (which takes ~10ms). And would penalize smaller model more then larger models.

> Thanks to @Tomeno & @Cahya for contributing benchmark numbers ofr their A1 and M2 Pro respectively

The above is done by downloading the respective model via `rwkv-cpp-node --setup`, and performing the `rwkv-cpp-node --dragon` benchmark. Which would give the following JSON at the end

```
... output of the benchmark ...
{"promptTime":178,"completionTime":109676,"totalTime":109854,"timePerPrompt":89,"timePerCompletion":109.676,"timePerFullPrompt":4.341463414634147,"promptPerSecond":11.235955056179776,"completionPerSecond":9.117765053430103,"fullPromptPerSecond":230.33707865168537}
```

timePerCompletion : is then extracted and used in the above table.

> Minor notes: 7B (Q5_1) uses ~ 7.1 GB ram, 7B (Q4_3) uses ~ 6.3 GB ram, making them ideal targets for 8GB ram systems

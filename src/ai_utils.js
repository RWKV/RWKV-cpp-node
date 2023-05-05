/**
 * Given the float array, get its max value.
 * @param {Float32Array} arr 
 * @returns {number}
 */
function getMaxFloat(arr) {
	let max = -Infinity;
	for (let i = 0; i < arr.length; i++) {
		if (arr[i] > max) max = arr[i];
	}
	return max;
}

/**
 * Implements the softmax function.
 * This modifies the input array.
 * 
 * @param {Float32Array} arr 
 * @returns 
 */
function softmax(arr) {
	// Get the max value
	const max = getMaxFloat(arr);

	// Subtract the max value from each element
	// and calculate the sum of exponents
	let sum = 0.0;
	for (let i = 0; i < arr.length; i++) {
		arr[i] = Math.exp(arr[i] - max);
		sum += arr[i];
	}

	// Divide each element by the sum
	for (let i = 0; i < arr.length; i++) {
		arr[i] = arr[i] / sum;
	}

	// Return the modified array
	return arr;
}

/**
 * sample_logits operation, used to decide on the next token
 * given the current logits output.
 * 
 * @param {Float32Array} logits - The logits output from the model
 * @param {number} temp - The temperature to use for sampling
 * @param {number} top_p - The top_p to use for sampling
 * 
 * @returns {Object} containing the token index, and the final logits
 */
function sampleLogits(logits, temp = 1.0, top_p = 1.0) {
	// Validate the logits buffer
	if (logits == null) {
		throw "Invalid logits buffer";
	}

	// If temp is 0.0, then we just return the max logit index
	if (temp <= 0.0) {
		return logits.indexOf(getMaxFloat(logits));
	}

	// Validate the top_p
	if (top_p < 0.0) {
		throw "Invalid top_p";
	}

	// Normalize temp, and top_p as float values
	temp = temp * 1.0;
	top_p = top_p * 1.0;

	// Get the logits size
	const logits_size = logits.length;

	// Create a new array to hold the scaled logits
	const scaled_logits = new Float32Array(logits_size);

	// Scale the logits by the temperature
	for (let i = 0; i < logits_size; i++) {
		scaled_logits[i] = logits[i] / temp;
	}

	// Apply softmax to obtain probabilities
	const probs = softmax(scaled_logits);
	
	// Change into a list of [index, prob] pairs
	let probPairs = [];
	for (let i = 0; i < probs.length; i++) {
		probPairs.push([i, probs[i]]);
	}

	// Sort the pairs by probability
	probPairs.sort((a, b) => b[1] - a[1]);

	// Calculate the cumulative probability
	let cumProb = 0.0;
	
	// Apply top_p filtering
	if (top_p < 1.0) {
		for (let i = 0; i < probPairs.length; i++) {
			cumProb += probPairs[i][1];

			// If we have reached the top_p threshold, then break
			if (cumProb >= top_p) {
				probPairs = probPairs.slice(0, i + 1);
				break;
			}
		}
	} else {
		// If top_p is 1.0, then we just use the full list
		for (let i = 0; i < probPairs.length; i++) {
			cumProb += probPairs[i][1];
		}
	}

	// Time to sample 
	let randProb = Math.random() * cumProb;

	// Find the index of the sampled token
	for(let i = 0; i < probPairs.length; i++) {
		randProb -= probPairs[i][1];
		if (randProb <= 0.0) {
			return {
				token: probPairs[i][0],
				logprobs: probPairs
			};
		}
	}

	// Out of bound? return the first index
	// (higest probability token)
	//
	// This should not happen unless an extream case 
	// of floating point accuracy error
	return {
		token: probPairs[0][0],
		logprobs: probPairs
	};
}

// Module exports
module.exports = {
	sampleLogits,
	softmax,
	getMaxFloat
}
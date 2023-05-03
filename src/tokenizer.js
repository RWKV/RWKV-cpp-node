const fs = require('fs');

const config = JSON.parse(fs.readFileSync('./src/20B_tokenizer.json'));

function tokenize(input) {
  // Normalize input string
  input = input.normalize(config.normalizer.type);

  // Apply pre-tokenizer
  let preTokenized = config.pre_tokenizer.type === 'ByteLevel'
    ? [...input]
    : config.pre_tokenizer.type(input);

  // Apply post-processor
  let processedTokens = config.post_processor.type === 'ByteLevel'
    ? preTokenized.map(token => ({ value: token, offsets: [0, 1] }))
    : config.post_processor.type(preTokenized);

  // Apply BPE encoding
  let encodedTokens = [];
  for (let i = 0; i < processedTokens.length; i++) {
    let subwords = applyBPE(processedTokens[i].value);
    let tokenOffsets = processedTokens[i].offsets;
    for (let j = 0; j < subwords.length; j++) {
      let token = subwords[j];
      let start = tokenOffsets[0];
      let end = tokenOffsets[0] + token.length;
      encodedTokens.push({ id: getTokenId(token), offsets: [start, end] });
      tokenOffsets[0] = end;
    }
  }

  // Apply padding and truncation
  if (config.padding !== null || config.truncation !== null) {
    let maxLength = config.truncation !== null ? config.truncation.max_length : Infinity;
    while (encodedTokens.length < maxLength) {
      encodedTokens.push({ id: getPaddingId() });
    }
    if (encodedTokens.length > maxLength) {
      encodedTokens = encodedTokens.slice(0, maxLength);
    }
  }

  return encodedTokens.map(token => token.id);
}

function applyBPE(token) {
    let subwords = [token];
    let pairFound = true;
    
    while (pairFound) {
      let newSubwords = [];
      pairFound = false;
  
      for (let subword of subwords) {
        let subwordFound = false;
        for (let merge of config.model.merges) {
          let [a, b] = merge.split(' ');
  
          let index = subword.indexOf(a + b);
          if (index >= 0) {
            let prefix = subword.slice(0, index);
            let suffix = subword.slice(index + a.length + b.length);
            
            newSubwords.push(...[prefix, a+b, suffix].filter(Boolean));
            subwordFound = true;
            pairFound = true;
            break;
          }
        }
        if (!subwordFound) {
          newSubwords.push(subword);
        }
      }
      subwords = newSubwords;
    }
  
    return subwords;
  }

function getTokenId(token) {
  // Search token in vocab
  let id = config.model.vocab[token];
  if (id === undefined) {
    // Handle unknown tokens
    id = config.model.unk_token;
    if (id === null) {
      id = config.model.vocab[config.model.unk_token];
    }
    if (id === undefined) {
      throw new Error(`Unknown token: ${token}`);
    }
  }
  return id;
}

function getPaddingId() {
  return config.model.vocab[config.added_tokens.find(token => token.special && token.content === '<|padding|>').content];
}

// Usage example
let input = "Hello world!";
let result = tokenize(input);
console.log(result);
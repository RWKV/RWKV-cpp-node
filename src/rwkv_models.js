module.exports = [
	{
		label: 'RWKV raven 1B5 v11 (Small, Fast, Dumb)',
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
		label: 'RWKV raven 7B v11 (Q8_0, multilingual)', 
		name: "raven_7b_v11_Q8_0_multilingual.bin",
		url: 'https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/Q8_0-RWKV-4-Raven-7B-v11-Eng49%25-Chn49%25-Jpn1%25-Other1%25-20230430-ctx8192.bin',
		sha256: 'ee4a6e7fbf9c2bd3558e4a92dbf16fd25d8599c7bef379ba91054981a8f665e0',
		size: 8681332157
	},
	{ 
		label: 'RWKV raven 14B v11 (Q8_0, Bigger, better)',
		name: "raven_14b_v11_Q8_0.bin",
		url: 'https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/Q8_0-RWKV-4-Raven-14B-v11x-Eng99%25-Other1%25-20230501-ctx8192.bin',
		sha256: '1cb56cd7784264a8f0ed2efd17a97f4866075915f2b1a8497c1abb826edade8b',
		size: 16374220069
	},
	{ 
		label: 'RWKV raven 14B v11 (Best results, slowest perf)',
		name: "raven_14b_v11x.bin",
		url: 'https://huggingface.co/datasets/picocreator/rwkv-4-cpp-quantize-bin/resolve/main/RWKV-4-Raven-14B-V11x.bin',
		sha256: 'f25f80555c840ad42b19411ca50788632eae9a714028b5d2431f6525d3296bce',
		size: 28301772069
	},
	{ 
		label: 'RWKV Pile 169M (Q8_0, lacks tuning, use for testing)',
		name: "rwkv_169M_pileplus_Q8_0.bin",
		url: 'https://huggingface.co/datasets/picocreator/rwkv-4-cpp-quantize-bin/resolve/main/RWKV-4-PilePlus-169M-Q8_0.bin',
		sha256: '82c2949f6f9261543b13cbd1409fd2069cd67d9e2ad031bb727bb0bd43527af1',
		size: 258391865
	}
]
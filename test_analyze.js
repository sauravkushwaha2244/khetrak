const fs = require('fs');
const metadata = JSON.parse(fs.readFileSync('public/model_metadata.json', 'utf8'));

// Test simpleHash logic
function simpleHash(s) { let h = 0; for (const c of s) h = ((h << 5) - h + c.charCodeAt(0)) | 0; return Math.abs(h); }

const predsClassName = "web site, website, internet site, site";
const hash = simpleHash(predsClassName);
let classes = metadata.classes;
let selectedCrop = 'all';

if (selectedCrop !== 'all') classes = classes.filter(c => c.crop.toLowerCase() === selectedCrop.toLowerCase());
const cls = classes[hash % classes.length];

console.log("hash:", hash);
console.log("length:", classes.length);
console.log("cls:", cls);

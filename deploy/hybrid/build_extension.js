// Copy only our two browser-safe modules. No official game code is packaged.
const fs=require('fs'),path=require('path');
const destination=path.resolve(__dirname,'../hybrid_extension');
for(const name of ['official_profile.js','arbiter.js','shortcut.js'])fs.copyFileSync(path.join(__dirname,name),path.join(destination,name));
console.log('Extension modules prepared: '+destination);

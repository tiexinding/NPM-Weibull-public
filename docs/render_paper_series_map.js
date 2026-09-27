// Render docs/paper_series_map.svg to docs/paper_series_map.png with sharp (npm install sharp).
const path = require('path');
const sharp = require('sharp');
const dir = __dirname;
sharp(path.join(dir, 'paper_series_map.svg'), { density: 144 })
  .png({ compressionLevel: 9 })
  .toFile(path.join(dir, 'paper_series_map.png'))
  .then(info => console.log('wrote paper_series_map.png', info.width + 'x' + info.height));

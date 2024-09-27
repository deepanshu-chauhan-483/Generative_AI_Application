import express from 'express';
import * as dotenv from 'dotenv';
import axios from 'axios';
import fs from 'fs';

dotenv.config();

const router = express.Router();

router.route('/').get((req, res) => {
  res.status(200).json({ message: 'Hello from DALL-E!' });
});

router.route('/').post(async (req, res) => {
  try {
    const { prompt } = req.body;

    const apiUrl = 'http://127.0.0.1:7860/sdapi/v1/txt2img';

    const requestBody = {
      prompt: prompt,
      negative_prompt: "",
      styles: [""],
      seed: -1,
      subseed: -1,
      subseed_strength: 0,
      seed_resize_from_h: -1,
      seed_resize_from_w: -1,
      batch_size: 1,
      n_iter: 1,
      steps: 50,
      cfg_scale: 7,
      width: 512,
      height: 512,
      restore_faces: true,
      tiling: true,
      do_not_save_samples: false,
      do_not_save_grid: false,
      eta: 0,
      denoising_strength: 0,
      s_min_uncond: 0,
      s_churn: 0,
      s_tmax: 0,
      s_tmin: 0,
      s_noise: 0,
      override_settings: {},
      override_settings_restore_afterwards: true,
      refiner_switch_at: 0,
      disable_extra_networks: false,
      comments: {},
      enable_hr: false,
      firstphase_width: 0,
      firstphase_height: 0,
      hr_scale: 2,
      hr_second_pass_steps: 0,
      hr_resize_x: 0,
      hr_resize_y: 0,
      hr_prompt: "",
      hr_negative_prompt: "",
      sampler_index: "Euler",
      script_args: [],
      send_images: true,
      save_images: false,
      alwayson_scripts: {}
    };

    const response = await axios.post(apiUrl, requestBody);

    console.log('API Response:', response.data);
    console.log('API Response:', response.data);

    const base64Image = response.data.image; // Adjust this based on the actual response structure
    const imageBuffer = Buffer.from(base64Image, 'base64');

    const imagePath = 'output.png';
    fs.writeFileSync(imagePath, imageBuffer);

    res.status(200).json({ photo: imagePath });
  } catch (error) {
    console.error(error);
    let errorMessage = 'Something went wrong';
    if (error.response && error.response.data) {
      try {
        errorMessage = JSON.stringify(error.response.data);
      } catch (e) {
        errorMessage = error.response.data;
      }
    }

    res.status(500).send(errorMessage);
  }
});

export default router;




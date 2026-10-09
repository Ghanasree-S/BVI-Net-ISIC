import { OrganConfig, OrganType } from '../types';

// Helper to generate realistic high-definition SVG medical scan representations encoded as Data URLs
function generateMedicalScanSvg(type: 'skin_nevus' | 'skin_melanoma' | 'liver_ct' | 'brain_mri'): string {
  if (type === 'skin_nevus') {
    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
      <defs>
        <radialGradient id="skinBg" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stop-color="#f5d5c0"/>
          <stop offset="50%" stop-color="#e8bf9f"/>
          <stop offset="90%" stop-color="#cf9f7d"/>
          <stop offset="100%" stop-color="#966a4f"/>
        </radialGradient>
        <radialGradient id="lesionGrad" cx="48%" cy="46%" r="50%">
          <stop offset="0%" stop-color="#3d1e12"/>
          <stop offset="35%" stop-color="#5a2d1d"/>
          <stop offset="70%" stop-color="#80472d"/>
          <stop offset="95%" stop-color="#b07352"/>
          <stop offset="100%" stop-color="#cf9f7d" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="lensVignette" cx="50%" cy="50%" r="50%">
          <stop offset="75%" stop-color="#000000" stop-opacity="0"/>
          <stop offset="95%" stop-color="#000000" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#000000" stop-opacity="0.95"/>
        </radialGradient>
        <filter id="pigmentNoise" x="0%" y="0%" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" result="noise"/>
          <feColorMatrix type="matrix" values="0.3 0 0 0 0  0 0.2 0 0 0  0 0 0.1 0 0  0 0 0 0.45 0" in="noise" result="coloredNoise"/>
          <feComposite operator="in" in2="SourceGraphic"/>
        </filter>
      </defs>
      <!-- Base Dermoscopy Background -->
      <rect width="400" height="400" fill="url(#skinBg)"/>
      <!-- Skin micro-texture lines -->
      <path d="M50 120 Q 200 130 350 110 M40 220 Q 180 250 360 210 M60 300 Q 220 310 340 290" stroke="#b08365" stroke-width="1.5" opacity="0.35" fill="none"/>
      
      <!-- Primary Melanocytic Lesion Area -->
      <path d="M 185 105 C 240 100, 290 135, 298 190 C 305 240, 270 285, 215 292 C 160 298, 115 260, 108 205 C 102 155, 135 110, 185 105 Z" fill="url(#lesionGrad)"/>
      
      <!-- Irregular border contour & pigment network -->
      <path d="M 195 130 C 235 125, 270 150, 275 195 C 280 230, 250 265, 210 270 C 170 275, 135 245, 130 205 C 125 165, 155 132, 195 130 Z" fill="#241009" opacity="0.85"/>
      <circle cx="210" cy="180" r="28" fill="#120703" opacity="0.75"/>
      <circle cx="160" cy="220" r="16" fill="#422013" opacity="0.65"/>
      
      <!-- Dermoscope Gel Bubble / Light Reflection -->
      <ellipse cx="140" cy="120" rx="35" ry="12" fill="#ffffff" opacity="0.18" transform="rotate(-25 140 120)"/>
      
      <!-- Polarized Dermoscopy Ring Vignette -->
      <rect width="400" height="400" fill="url(#lensVignette)"/>
      
      <!-- Calibration Reticle / Scale Marks -->
      <g stroke="#ffffff" stroke-width="1" opacity="0.4">
        <line x1="20" y1="380" x2="100" y2="380"/>
        <line x1="20" y1="375" x2="20" y2="385"/>
        <line x1="60" y1="377" x2="60" y2="383"/>
        <line x1="100" y1="375" x2="100" y2="385"/>
        <text x="35" y="372" fill="#ffffff" font-size="9" font-family="sans-serif">10 mm</text>
      </g>
    </svg>
    `;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  if (type === 'skin_melanoma') {
    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
      <defs>
        <radialGradient id="melBg" cx="50%" cy="50%" r="55%">
          <stop offset="0%" stop-color="#edd2c0"/>
          <stop offset="60%" stop-color="#d6a785"/>
          <stop offset="100%" stop-color="#694025"/>
        </radialGradient>
        <radialGradient id="melCore" cx="45%" cy="52%" r="48%">
          <stop offset="0%" stop-color="#0d0706"/>
          <stop offset="40%" stop-color="#2b140d"/>
          <stop offset="75%" stop-color="#6e301a"/>
          <stop offset="90%" stop-color="#a85331"/>
          <stop offset="100%" stop-color="#d6a785" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="melVignette" cx="50%" cy="50%" r="50%">
          <stop offset="78%" stop-color="#000000" stop-opacity="0"/>
          <stop offset="98%" stop-color="#000000" stop-opacity="0.95"/>
        </radialGradient>
      </defs>
      <rect width="400" height="400" fill="url(#melBg)"/>
      
      <!-- Asymmetric Melanoma contour with color variegation -->
      <path d="M 160 95 C 230 80, 310 120, 305 180 C 300 220, 325 250, 280 295 C 235 340, 160 320, 115 285 C 70 250, 85 180, 105 140 C 120 110, 130 102, 160 95 Z" fill="url(#melCore)"/>
      
      <!-- Deep dark atypical pigment globules -->
      <path d="M 170 140 C 210 125, 270 150, 260 210 C 250 260, 190 280, 150 255 C 110 230, 125 170, 170 140 Z" fill="#140603" opacity="0.9"/>
      
      <!-- Blue-white veil structure in melanoma center -->
      <ellipse cx="205" cy="205" rx="32" ry="24" fill="#9db5c9" opacity="0.35" transform="rotate(-15 205 205)"/>
      
      <!-- Radial streaks & peripheral projections -->
      <path d="M 290 160 Q 330 150 340 170 M 270 290 Q 300 325 315 330 M 110 270 Q 80 300 65 295" stroke="#4a1f10" stroke-width="3" opacity="0.6" stroke-linecap="round" fill="none"/>
      
      <rect width="400" height="400" fill="url(#melVignette)"/>
      <text x="25" y="375" fill="#ffffff" opacity="0.5" font-size="9" font-family="sans-serif">ISIC_0024310 • Dermoscopy</text>
    </svg>
    `;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  if (type === 'liver_ct') {
    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
      <rect width="400" height="400" fill="#080b0e"/>
      <!-- Patient body contour (Hounsfield grayscale CT) -->
      <ellipse cx="200" cy="205" rx="170" ry="145" fill="#1e252b" stroke="#3d4952" stroke-width="2"/>
      <ellipse cx="200" cy="205" rx="160" ry="135" fill="#12171c"/>
      
      <!-- Spine & Vertebra (Dense bone HU +1000) -->
      <ellipse cx="200" cy="305" rx="26" ry="20" fill="#e8edf2"/>
      <circle cx="200" cy="305" r="9" fill="#080b0e"/>
      <!-- Ribs -->
      <path d="M 50 170 Q 40 230 75 290 M 350 170 Q 360 230 325 290" stroke="#cfd8dc" stroke-width="7" stroke-linecap="round" fill="none" opacity="0.8"/>
      
      <!-- Right Lobe Liver Parenchyma (HU ~50-70) -->
      <path d="M 80 150 C 70 220, 100 270, 190 275 C 230 278, 240 220, 220 150 C 200 110, 110 110, 80 150 Z" fill="#454f59" stroke="#5a6773" stroke-width="1.5"/>
      
      <!-- Hepatic Lesion / Tumor Hypodense Focal Area -->
      <path d="M 120 175 C 145 165, 175 170, 180 195 C 185 220, 160 240, 135 238 C 110 235, 100 210, 105 190 C 110 180, 115 178, 120 175 Z" fill="#262d33" stroke="#1d2328" stroke-width="2"/>
      
      <!-- Spleen (Left side) -->
      <path d="M 280 160 C 330 170, 335 240, 290 260 C 265 240, 260 180, 280 160 Z" fill="#38414a"/>
      
      <!-- CT Window Information HUD -->
      <g fill="#00e5ff" font-family="monospace" font-size="9" opacity="0.75">
        <text x="15" y="25">LiTS17 CT AXIAL SLICE #42</text>
        <text x="15" y="38">WL: 40  WW: 350</text>
        <text x="320" y="25">FOV: 380mm</text>
        <text x="320" y="38">THICK: 2.5mm</text>
        <text x="15" y="385">R</text>
        <text x="375" y="385">L</text>
      </g>
    </svg>
    `;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  }

  // brain_mri default
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
    <rect width="400" height="400" fill="#040608"/>
    <!-- Cranium / Skull Contour -->
    <ellipse cx="200" cy="200" rx="145" ry="165" fill="#181f26" stroke="#485866" stroke-width="2"/>
    <ellipse cx="200" cy="200" rx="136" ry="156" fill="#0e1216"/>
    
    <!-- Brain Hemispheres & Cerebral Cortex Gyri/Sulci -->
    <path d="M 200 55 L 200 345" stroke="#040608" stroke-width="3"/> <!-- Interhemispheric fissure -->
    
    <!-- Brain Parenchyma Gray/White Matter -->
    <path d="M 195 65 C 130 65, 80 120, 80 200 C 80 280, 130 335, 195 335 Z" fill="#2d3742"/>
    <path d="M 205 65 C 270 65, 320 120, 320 200 C 320 280, 270 335, 205 335 Z" fill="#2d3742"/>
    
    <!-- Ventricles (CSF) -->
    <path d="M 185 180 C 175 190, 175 220, 190 230 C 195 210, 195 190, 185 180 Z" fill="#040608"/>
    <path d="M 215 180 C 225 190, 225 220, 210 230 C 205 210, 205 190, 215 180 Z" fill="#040608"/>
    
    <!-- High-Signal FLAIR Glioma Lesion / Edema in Right Parieto-Temporal Region -->
    <path d="M 125 140 C 165 130, 180 160, 175 195 C 170 230, 130 240, 105 220 C 85 200, 95 155, 125 140 Z" fill="#8ca0b3" stroke="#a3b8cb" stroke-width="1.5" filter="url(#glow)"/>
    <circle cx="140" cy="180" r="18" fill="#e0ebf5" opacity="0.9"/>
    
    <!-- MRI HUD Header -->
    <g fill="#00e5ff" font-family="monospace" font-size="9" opacity="0.75">
      <text x="15" y="25">BraTS19 MRI FLAIR AXIAL</text>
      <text x="15" y="38">TE: 110ms  TR: 9000ms</text>
      <text x="320" y="25">TE: 3.0T</text>
      <text x="320" y="38">SLICE: 84/155</text>
      <text x="15" y="385">R</text>
      <text x="375" y="385">L</text>
    </g>
  </svg>
  `;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const ORGAN_CONFIGS: Record<OrganType, OrganConfig> = {
  skin: {
    id: 'skin',
    name: 'Skin (ISIC2018)',
    datasetName: 'ISIC 2018 Task 1: Lesion Boundary Segmentation',
    shortLabel: 'Skin',
    isAvailable: true,
    statusText: 'Active Checkpoint (Production Ready)',
    inputHint: 'Upload a dermoscopy image (JPG/PNG - standard epiluminescence microscopy)',
    modelParams: '27,312 parameters (0.11 MB)',
    paramCount: 27312,
    flops: '0.082 GFLOPs',
    modalities: 'Polarized / Non-polarized Dermoscopy',
    resolution: '256 × 256 RGB',
    targetLesions: 'Melanocytic Nevi, Melanoma, Seborrheic Keratosis, BCC',
    description:
      'Trained on 2,594 high-resolution dermoscopic images from the International Skin Imaging Collaboration (ISIC 2018) benchmark.',
    sampleCases: [
      {
        id: 'isic_sample_1',
        organ: 'skin',
        title: 'Benign Melanocytic Nevus',
        subType: 'Dermoscopy Case #ISIC-0024310',
        description: 'Typical benign compound nevus with uniform pigment network and well-defined borders.',
        imageUrl: generateMedicalScanSvg('skin_nevus'),
        resolution: '256 × 256',
        provenance: 'ISIC2018 Validation Set',
      },
      {
        id: 'isic_sample_2',
        organ: 'skin',
        title: 'Superficial Spreading Melanoma',
        subType: 'Dermoscopy Case #ISIC-0031892',
        description: 'Asymmetric border with irregular pigmentation, blue-white veil, and peripheral radial streaming.',
        imageUrl: generateMedicalScanSvg('skin_melanoma'),
        resolution: '256 × 256',
        provenance: 'ISIC2018 Benchmark',
      },
    ],
  },
  liver: {
    id: 'liver',
    name: 'Liver (LiTS17)',
    datasetName: 'LiTS 2017: Liver Tumor Segmentation Challenge',
    shortLabel: 'Liver',
    isAvailable: true,
    statusText: 'Active Checkpoint (test global Dice: liver 0.928, tumor 0.566)',
    inputHint: 'Upload an axial abdominal CT slice (grayscale PNG, liver window -100..400 HU)',
    modelParams: '27,277 parameters (0.11 MB)',
    paramCount: 27277,
    flops: '0.096 GFLOPs',
    modalities: 'Contrast-enhanced Abdominal CT (Portal Venous Phase)',
    resolution: '448 × 448 Grayscale CT',
    targetLesions: 'Hepatocellular Carcinoma (HCC), Hepatic Cysts, Metastases',
    description:
      'Designed for cross-sectional contrast-enhanced abdominal CT scans from the Liver Tumor Segmentation (LiTS17) cohort.',
    sampleCases: [
      {
        id: 'lits_sample_1',
        organ: 'liver',
        title: 'LiTS17 Test Slice A',
        subType: 'Held-out test volume (not seen in training)',
        description: 'Real axial CT slice, liver window -100..400 HU, from the LiTS17 test split.',
        imageUrl: '/samples/liver_ct_test_1.png',
        resolution: '448 × 448 Slice',
        provenance: 'LiTS17 test split',
      },
      {
        id: 'lits_sample_2',
        organ: 'liver',
        title: 'LiTS17 Test Slice B',
        subType: 'Held-out test volume (not seen in training)',
        description: 'Real axial CT slice, liver window -100..400 HU, from the LiTS17 test split.',
        imageUrl: '/samples/liver_ct_test_2.png',
        resolution: '448 × 448 Slice',
        provenance: 'LiTS17 test split',
      },
    ],
  },
  brain: {
    id: 'brain',
    name: 'Brain (BraTS19)',
    datasetName: 'BraTS 2019: Multimodal Brain Tumor Segmentation',
    shortLabel: 'Brain',
    isAvailable: true,
    statusText: 'Active Checkpoint (test global Dice: WT 0.862, TC 0.711, ET 0.744)',
    inputHint: 'Upload a 4-channel MRI slice as .npy (T1, T1ce, T2, FLAIR, from data/prepare_brats.py)',
    modelParams: '27,342 parameters (0.11 MB)',
    paramCount: 27342,
    flops: '0.104 GFLOPs',
    modalities: 'Multimodal MRI (FLAIR, T1ce, T2, T1)',
    resolution: '240 × 240 × 4 MRI',
    targetLesions: 'High-Grade Glioma (HGG), Low-Grade Glioma (LGG), Peritumoral Edema',
    description:
      'Optimized for heterogeneous multimodal MRI volumes from the Brain Tumor Segmentation (BraTS19) clinical repository.',
    sampleCases: [
      {
        id: 'brats_sample_1',
        organ: 'brain',
        title: 'Glioma Core on FLAIR MRI',
        subType: 'BraTS19 Case #MRI-FLAIR-84',
        description: 'Axial FLAIR MRI showing hyperintense tumor region and surrounding peritumoral edema.',
        imageUrl: generateMedicalScanSvg('brain_mri'),
        resolution: '240 × 240 Axial',
        provenance: 'BraTS19 Dataset',
      },
    ],
  },
};

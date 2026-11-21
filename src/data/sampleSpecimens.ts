import type { SampleSpecimen } from '../types/diagnostics';

export const SAMPLE_SPECIMENS: SampleSpecimen[] = [
  {
    id: 'tomato-early-blight',
    label: 'Tomato · Early Blight',
    crop: 'Tomato',
    expectedClass: 'Tomato___Early_blight',
    imageUrl: '/src/assets/images/sample_tomato_early_blight_1795288685343.jpg',
    specimenCode: 'PV-TOM-029',
    notes: 'Concentric target-like dark brown lesions with chlorotic halos (Alternaria solani).',
  },
  {
    id: 'apple-scab',
    label: 'Apple · Apple Scab',
    crop: 'Apple',
    expectedClass: 'Apple___Apple_scab',
    imageUrl: '/src/assets/images/sample_apple_scab_1795288683759.jpg',
    specimenCode: 'PV-APP-001',
    notes: 'Olive-brown velvety scab blotches distributed along primary and secondary veins.',
  },
  {
    id: 'grape-black-rot',
    label: 'Grape · Black Rot',
    crop: 'Grape',
    expectedClass: 'Grape___Black_rot',
    imageUrl: '/src/assets/images/sample_grape_black_rot_1795288685586.jpg',
    specimenCode: 'PV-GRP-012',
    notes: 'Circular tan-brown necrotic spots framed by dark margins and central pycnidia.',
  },
  {
    id: 'corn-healthy',
    label: 'Corn · Healthy Control',
    crop: 'Corn (Maize)',
    expectedClass: 'Corn_(maize)___healthy',
    imageUrl: '/src/assets/images/sample_corn_healthy_1795288685828.jpg',
    specimenCode: 'PV-CRN-011',
    notes: 'Disease-free control specimen with uniform chlorophyll and crisp parallel venation.',
  },
];

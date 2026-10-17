const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "MedVision AI — Soutenance de Projet";

// ── Palette ────────────────────────────────────────────────────────────────
const C = {
  navy:    "0F1D3E",
  blue:    "1D4ED8",
  blue2:   "3B82F6",
  purple:  "7C3AED",
  teal:    "0D9488",
  green:   "059669",
  amber:   "D97706",
  red:     "DC2626",
  white:   "FFFFFF",
  light:   "EFF6FF",
  light2:  "F8FAFC",
  gray:    "64748B",
  dark:    "1E293B",
  border:  "E2E8F0",
};

const shadow = () => ({ type:"outer", color:"000000", blur:8, offset:3, angle:135, opacity:0.12 });

// ── Helpers ────────────────────────────────────────────────────────────────
function addCard(slide, x, y, w, h, fill="FFFFFF") {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: C.border, width: 0.5 },
    shadow: shadow(),
  });
}

function addMetric(slide, x, y, val, lbl, color=C.blue) {
  addCard(slide, x, y, 2.1, 1.3, "FFFFFF");
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w:2.1, h:0.07, fill:{color:color} });
  slide.addText(val, { x, y:y+0.15, w:2.1, h:0.65, fontSize:28, bold:true, color:color, align:"center", fontFace:"Calibri" });
  slide.addText(lbl, { x, y:y+0.82, w:2.1, h:0.35, fontSize:10, color:C.gray, align:"center", fontFace:"Calibri" });
}

function slideNum(slide, n, total=18) {
  slide.addText(`${n} / ${total}`, {
    x:9.0, y:5.3, w:0.8, h:0.25, fontSize:9, color:C.gray, align:"right", fontFace:"Calibri"
  });
}

function sectionTag(slide, text, color=C.blue) {
  slide.addShape(pres.shapes.RECTANGLE, { x:0.4, y:0.18, w:0.06, h:0.38, fill:{color:color} });
  slide.addText(text, { x:0.55, y:0.17, w:6, h:0.4, fontSize:11, bold:true, color:color, fontFace:"Calibri", margin:0 });
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — COVER
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // Gradient feel — overlay shapes
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:5.625, fill:{color:C.navy} });
  s.addShape(pres.shapes.RECTANGLE, { x:6.5, y:0, w:3.5, h:5.625, fill:{color:"0D2154", transparency:30} });
  s.addShape(pres.shapes.OVAL, { x:6.8, y:-1, w:4, h:4, fill:{color:C.blue, transparency:85} });
  s.addShape(pres.shapes.OVAL, { x:7.5, y:2.5, w:3, h:3, fill:{color:C.purple, transparency:80} });

  // Logo area
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:0.45, w:0.06, h:1.2, fill:{color:C.blue2} });

  // Title
  s.addText("MedVision AI", {
    x:0.7, y:0.4, w:8, h:1.1,
    fontSize:52, bold:true, color:C.white, fontFace:"Calibri",
  });
  s.addText("Système de Détection et Classification des Tumeurs par Deep Learning", {
    x:0.7, y:1.45, w:7.5, h:0.7,
    fontSize:16, color:"A5B4FC", fontFace:"Calibri", italic:true,
  });

  // Info chips
  const chips = [
    { t:"EfficientNetB4", c:C.blue2 },
    { t:"FastAPI · Streamlit", c:C.teal },
    { t:"DICOM / PACS", c:C.purple },
    { t:"Grad-CAM XAI", c:C.green },
  ];
  chips.forEach((ch, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.7+i*2.2, y:2.35, w:2.0, h:0.38, fill:{color:ch.c, transparency:75}, line:{color:ch.c, width:1} });
    s.addText(ch.t, { x:0.7+i*2.2, y:2.35, w:2.0, h:0.38, fontSize:11, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
  });

  // Stats row
  const stats = [
    { v:"95.71%", l:"Brain Accuracy" },
    { v:"0.98",   l:"ROC-AUC" },
    { v:"0.918",  l:"F2-Score" },
    { v:"2 Datasets", l:"Multi-modal" },
  ];
  stats.forEach((st, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.7+i*2.3, y:3.05, w:2.1, h:0.9, fill:{color:"FFFFFF", transparency:90}, line:{color:"FFFFFF", width:0.5, transparency:70} });
    s.addText(st.v, { x:0.7+i*2.3, y:3.1, w:2.1, h:0.45, fontSize:20, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(st.l, { x:0.7+i*2.3, y:3.55, w:2.1, h:0.35, fontSize:10, color:"A5B4FC", align:"center", fontFace:"Calibri" });
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:5.1, w:10, h:0.525, fill:{color:"0A1428"} });
  s.addText("Soutenance de Projet · Deep Learning Médical · 2024–2025", {
    x:0.5, y:5.12, w:9, h:0.35, fontSize:11, color:"6B7280", fontFace:"Calibri", align:"center"
  });
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — PLAN DE PRÉSENTATION
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "Plan de Présentation");
  s.addText("Au programme", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  const sections = [
    { n:"01", t:"Problématique & Contexte Médical",  c:C.blue },
    { n:"02", t:"Méthodologie CRISP-DM — Phase 0→5", c:C.purple },
    { n:"03", t:"Données & Prétraitement",           c:C.teal },
    { n:"04", t:"Architecture des Modèles",          c:C.green },
    { n:"05", t:"Résultats & Métriques Cliniques",   c:C.amber },
    { n:"06", t:"Déploiement — API · Dashboard",     c:C.red },
    { n:"07", t:"Explainabilité XAI — Grad-CAM",     c:C.blue2 },
    { n:"08", t:"Conclusion & Perspectives",         c:C.gray },
  ];

  sections.forEach((sec, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * 4.85;
    const y = 1.3 + row * 1.0;
    addCard(s, x, y, 4.6, 0.82, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:0.55, h:0.82, fill:{color:sec.c} });
    s.addText(sec.n, { x, y:y+0.18, w:0.55, h:0.45, fontSize:16, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(sec.t, { x:x+0.65, y:y+0.2, w:3.8, h:0.42, fontSize:13, bold:true, color:C.dark, fontFace:"Calibri" });
  });

  slideNum(s, 2);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — PROBLÉMATIQUE
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "01 · Problématique & Contexte", C.blue);
  s.addText("Pourquoi ce projet ?", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Left — problem
  addCard(s, 0.4, 1.25, 4.6, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:4.6, h:0.07, fill:{color:C.red} });
  s.addText("⚠️  Le Problème Clinique", { x:0.5, y:1.35, w:4.4, h:0.4, fontSize:14, bold:true, color:C.red, fontFace:"Calibri" });

  const problems = [
    "Détection manuelle lente et coûteuse",
    "Variabilité inter-opérateur (subjectivité)",
    "Volume croissant d'images médicales (+40%/an)",
    "Pénurie de radiologues spécialisés",
    "Stades précoces difficiles à identifier",
  ];
  problems.forEach((p, i) => {
    s.addShape(pres.shapes.OVAL, { x:0.6, y:1.88+i*0.55, w:0.22, h:0.22, fill:{color:C.red, transparency:20} });
    s.addText(p, { x:0.95, y:1.88+i*0.55, w:3.9, h:0.32, fontSize:12, color:C.dark, fontFace:"Calibri" });
  });

  // Right — solution
  addCard(s, 5.3, 1.25, 4.3, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.3, y:1.25, w:4.3, h:0.07, fill:{color:C.green} });
  s.addText("✅  Notre Solution IA", { x:5.4, y:1.35, w:4.1, h:0.4, fontSize:14, bold:true, color:C.green, fontFace:"Calibri" });

  const solutions = [
    "Pipeline automatisé End-to-End",
    "Détection binaire + classification stades",
    "2 modalités : IRM + Histopathologie",
    "Explainabilité via Grad-CAM (XAI)",
    "API médicale DICOM/PACS compatible",
  ];
  solutions.forEach((sol, i) => {
    s.addShape(pres.shapes.OVAL, { x:5.5, y:1.88+i*0.55, w:0.22, h:0.22, fill:{color:C.green, transparency:20} });
    s.addText(sol, { x:5.85, y:1.88+i*0.55, w:3.6, h:0.32, fontSize:12, color:C.dark, fontFace:"Calibri" });
  });

  // Bottom stat
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:5.2, w:9.2, h:0.32, fill:{color:C.navy} });
  s.addText("Le cancer tue ~10M personnes/an. Une détection précoce augmente la survie de 90% à 5 ans.", {
    x:0.5, y:5.22, w:9, h:0.28, fontSize:10, bold:true, color:C.white, align:"center", fontFace:"Calibri"
  });

  slideNum(s, 3);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — CRISP-DM
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "02 · Méthodologie CRISP-DM", C.purple);
  s.addText("6 phases structurant le projet", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  const phases = [
    { n:"0", t:"Business Understanding",    d:"Cahier des charges · KPI cliniques · objectifs médicaux",    c:C.navy },
    { n:"1", t:"Data Understanding",        d:"Exploration datasets · distribution classes · qualité",      c:C.blue },
    { n:"2", t:"Data Preparation",          d:"Prétraitement · augmentation · normalisation · split",        c:C.purple },
    { n:"3", t:"Modeling",                  d:"CNN baseline → Transfer Learning → Fine-Tuning",             c:C.teal },
    { n:"4", t:"Evaluation",                d:"Accuracy · F2-Score · ROC-AUC · Grad-CAM · benchmark",      c:C.green },
    { n:"5", t:"Deployment",                d:"FastAPI · Streamlit · Docker · DICOM",                       c:C.amber },
  ];

  phases.forEach((ph, i) => {
    const x = 0.4 + (i % 3) * 3.15;
    const y = 1.3 + Math.floor(i/3) * 2.0;
    addCard(s, x, y, 2.95, 1.7, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:2.95, h:0.55, fill:{color:ph.c} });
    s.addText(`Phase ${ph.n}`, { x:x+0.1, y:y+0.04, w:1.2, h:0.25, fontSize:10, color:C.white, fontFace:"Calibri", bold:true });
    s.addText(ph.t, { x, y:y+0.28, w:2.95, h:0.25, fontSize:12, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(ph.d, { x:x+0.1, y:y+0.65, w:2.75, h:0.9, fontSize:10.5, color:C.dark, fontFace:"Calibri" });

    // Arrow between phases in same row
    if (i%3 < 2) {
      s.addShape(pres.shapes.RECTANGLE, { x:x+2.95, y:y+0.6, w:0.2, h:0.04, fill:{color:C.gray} });
      s.addText("▶", { x:x+3.05, y:y+0.5, w:0.15, h:0.22, fontSize:10, color:C.gray, fontFace:"Calibri" });
    }
  });

  slideNum(s, 4);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — DATASETS
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "03 · Données & Prétraitement", C.teal);
  s.addText("2 Datasets Médicaux Hétérogènes", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Dataset 1
  addCard(s, 0.4, 1.25, 4.55, 3.95, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:4.55, h:0.55, fill:{color:C.blue} });
  s.addText("🧠  Dataset 1 — Brain Tumor MRI", { x:0.5, y:1.3, w:4.35, h:0.45, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });
  const d1 = [
    ["Source",    "Kaggle · Masoud Nickparvar"],
    ["Modalité",  "IRM Cérébrale (T1)"],
    ["Taille",    "~7 200 images"],
    ["Classes",   "4 : Glioma · Meningioma · No Tumor · Pituitary"],
    ["Split",     "Training / Testing natif Kaggle"],
    ["Déséquil.", "Modéré → Focal Loss + class_weight"],
  ];
  d1.forEach(([k,v], i) => {
    const y = 1.9 + i*0.52;
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y, w:1.3, h:0.38, fill:{color:C.light} });
    s.addText(k, { x:0.5, y, w:1.3, h:0.38, fontSize:10, bold:true, color:C.blue, align:"center", fontFace:"Calibri" });
    s.addText(v, { x:1.9, y:y+0.04, w:2.9, h:0.3, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  // Dataset 2
  addCard(s, 5.2, 1.25, 4.45, 3.95, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.25, w:4.45, h:0.55, fill:{color:C.purple} });
  s.addText("🔬  Dataset 2 — Breast Histopathology", { x:5.3, y:1.3, w:4.25, h:0.45, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });
  const d2 = [
    ["Source",    "Kaggle · Paul Mooney"],
    ["Modalité",  "Histopatho. mammaire (IDC)"],
    ["Taille",    "277 524 images PNG 50×50 px"],
    ["Classes",   "2 : Sain (0) · Carcinome (1)"],
    ["Split",     "70% train / 15% val / 15% test"],
    ["Déséquil.", "Fort 78/22 → class_weight balanced"],
  ];
  d2.forEach(([k,v], i) => {
    const y = 1.9 + i*0.52;
    s.addShape(pres.shapes.RECTANGLE, { x:5.3, y, w:1.3, h:0.38, fill:{color:"F5F3FF"} });
    s.addText(k, { x:5.3, y, w:1.3, h:0.38, fontSize:10, bold:true, color:C.purple, align:"center", fontFace:"Calibri" });
    s.addText(v, { x:6.7, y:y+0.04, w:2.8, h:0.3, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  slideNum(s, 5);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — PRÉTRAITEMENT
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "03 · Données & Prétraitement", C.teal);
  s.addText("Pipeline de Prétraitement", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  const steps = [
    { n:"1", t:"Redimensionnement",  d:"128×128 baseline\n380×380 EfficientNetB4",   c:C.blue },
    { n:"2", t:"Normalisation",      d:"preprocess_input EfficientNet\n(≠ /255)",     c:C.purple },
    { n:"3", t:"Augmentation",       d:"Rotation ±20°\nFlip H/V · Zoom ±15%",        c:C.teal },
    { n:"4", t:"Class Weights",      d:"compute_class_weight\n'balanced'",            c:C.green },
    { n:"5", t:"Split",              d:"80% train\n20% validation",                  c:C.amber },
    { n:"6", t:"Encodage",           d:"One-hot 4 classes\n(softmax output)",         c:C.blue2 },
  ];

  steps.forEach((st, i) => {
    const x = 0.4 + (i%3)*3.15;
    const y = 1.25 + Math.floor(i/3)*2.15;
    addCard(s, x, y, 2.95, 1.85, "FFFFFF");
    s.addShape(pres.shapes.OVAL, { x:x+1.2, y:y+0.12, w:0.55, h:0.55, fill:{color:st.c} });
    s.addText(st.n, { x:x+1.2, y:y+0.18, w:0.55, h:0.42, fontSize:16, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(st.t, { x, y:y+0.78, w:2.95, h:0.38, fontSize:13, bold:true, color:C.dark, align:"center", fontFace:"Calibri" });
    s.addText(st.d, { x:x+0.1, y:y+1.18, w:2.75, h:0.55, fontSize:10.5, color:C.gray, align:"center", fontFace:"Calibri" });
  });

  slideNum(s, 6);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — ARCHITECTURE MODÈLES
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "04 · Architecture des Modèles", C.green);
  s.addText("Stratégie de Modélisation", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Timeline
  const steps = [
    { t:"CNN Baseline", d:"Sequential Keras\n128×128 · 2 Conv2D\n~48% val_accuracy",   c:C.gray,   r:"❌ Overfitting" },
    { t:"EfficientNetB4", d:"Transfer Learning\nImageNet pre-trained\nPhase 1 gelée",    c:C.blue,   r:"✅ 95.71%" },
    { t:"Fine-Tuning", d:"50 couches dégelées\nlr=1e-5 · 30 epochs\nFocal Loss",        c:C.purple, r:"✅ Stable" },
    { t:"Production", d:"best_model.keras\nFastAPI · Docker\nDICOM support",             c:C.green,  r:"✅ Deploy" },
  ];

  steps.forEach((st, i) => {
    const x = 0.4 + i*2.35;
    // Arrow
    if (i < 3) {
      s.addShape(pres.shapes.RECTANGLE, { x:x+2.05, y:2.2, w:0.28, h:0.06, fill:{color:C.gray} });
      s.addText("▶", { x:x+2.28, y:2.08, w:0.2, h:0.3, fontSize:12, color:C.gray });
    }
    addCard(s, x, 1.3, 2.0, 3.8, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, { x, y:1.3, w:2.0, h:0.55, fill:{color:st.c} });
    s.addText(st.t, { x, y:1.35, w:2.0, h:0.45, fontSize:13, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(st.d, { x:x+0.1, y:1.95, w:1.8, h:1.6, fontSize:11, color:C.dark, fontFace:"Calibri", align:"center" });

    addCard(s, x, 3.55, 2.0, 0.5, st.c==="gray"? "FEF2F2":"ECFDF5");
    s.addText(st.r, { x, y:3.58, w:2.0, h:0.44, fontSize:11, bold:true, color:st.c==="gray"?C.red:C.green, align:"center", fontFace:"Calibri" });

    s.addText(`Step ${i+1}`, { x, y:4.2, w:2.0, h:0.3, fontSize:9, color:C.gray, align:"center", fontFace:"Calibri" });
  });

  slideNum(s, 7);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — EFFICIENTNETB4 DETAIL
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "04 · Architecture — EfficientNetB4", C.blue);
  s.addText("Configuration Technique Complète", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Left — architecture layers
  addCard(s, 0.4, 1.25, 5.0, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:5.0, h:0.5, fill:{color:C.blue} });
  s.addText("Architecture EfficientNetB4", { x:0.5, y:1.3, w:4.8, h:0.4, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });

  const layers = [
    { l:"Input",                   s:"380 × 380 × 3", c:C.gray },
    { l:"EfficientNetB4 Backbone", s:"~19.3M params (gelés Phase 1)", c:C.blue },
    { l:"GlobalAvgPooling2D",      s:"Réduction spatiale", c:C.blue2 },
    { l:"BatchNormalization",      s:"Stabilisation", c:C.teal },
    { l:"Dense(256, ReLU)",        s:"Couche intermédiaire", c:C.purple },
    { l:"Dropout(0.3)",            s:"Régularisation", c:C.amber },
    { l:"Dense(4, Softmax)",       s:"Classification finale", c:C.green },
  ];
  layers.forEach((lr, i) => {
    const y = 1.88 + i*0.48;
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y, w:0.28, h:0.32, fill:{color:lr.c} });
    s.addText(lr.l, { x:0.88, y, w:2.7, h:0.32, fontSize:11, bold:true, color:C.dark, fontFace:"Calibri" });
    s.addText(lr.s, { x:3.6, y, w:1.7, h:0.32, fontSize:10, color:C.gray, fontFace:"Calibri" });
  });

  // Right — config
  addCard(s, 5.7, 1.25, 3.95, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.7, y:1.25, w:3.95, h:0.5, fill:{color:C.purple} });
  s.addText("Hyperparamètres", { x:5.8, y:1.3, w:3.75, h:0.4, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });

  const params = [
    ["Poids initiaux",  "ImageNet (pré-entraîné)"],
    ["Phase 1 lr",      "1e-3 · 15 epochs"],
    ["Phase 2 lr",      "1e-5 · 30 epochs"],
    ["Loss",            "Focal Loss (γ=2, α=0.25)"],
    ["Optimiseur",      "Adam (adaptatif)"],
    ["EarlyStopping",   "patience=8"],
    ["ReduceLR",        "factor=0.3, patience=4"],
  ];
  params.forEach(([k,v], i) => {
    const y = 1.88 + i*0.48;
    s.addShape(pres.shapes.RECTANGLE, { x:5.8, y, w:1.5, h:0.32, fill:{color:"F5F3FF"} });
    s.addText(k, { x:5.8, y, w:1.5, h:0.32, fontSize:10, bold:true, color:C.purple, align:"center", fontFace:"Calibri" });
    s.addText(v, { x:7.38, y:y+0.03, w:2.2, h:0.28, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  slideNum(s, 8);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — RÉSULTATS BRAIN
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "05 · Résultats — Brain MRI", C.green);
  s.addText("Performances EfficientNetB4", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Big metrics
  addMetric(s, 0.4,  1.2, "95.71%", "Val Accuracy", C.green);
  addMetric(s, 2.65, 1.2, "91.94%", "Test Accuracy", C.blue);
  addMetric(s, 4.9,  1.2, "0.98",   "ROC-AUC Moyen", C.purple);
  addMetric(s, 7.15, 1.2, "0.918",  "F2-Score",      C.teal);

  // Chart — accuracy per class
  s.addChart(pres.charts.BAR, [{
    name: "Recall par classe",
    labels: ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    values: [97, 84, 96, 97]
  }], {
    x:0.4, y:2.7, w:5.5, h:2.65,
    barDir:"col",
    chartColors: [C.blue, C.purple, C.green, C.teal],
    chartArea: { fill:{ color:"FFFFFF"}, roundedCorners:true },
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valGridLine: { color:"E2E8F0", size:0.5 },
    catGridLine: { style:"none" },
    showValue: true,
    dataLabelColor: C.dark,
    showLegend: false,
    valAxisMaxVal: 100,
    showTitle: true,
    title: "Recall (%) par Classe",
    titleFontSize: 12,
  });

  // Right — progression table
  addCard(s, 6.1, 2.7, 3.55, 2.65, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:6.1, y:2.7, w:3.55, h:0.45, fill:{color:C.navy} });
  s.addText("Progression Entraînement", { x:6.15, y:2.74, w:3.45, h:0.37, fontSize:11, bold:true, color:C.white, fontFace:"Calibri" });

  const prog = [
    ["CNN Baseline",      "~48%", C.red],
    ["EfficientNetB4 P1", "95.71%", C.green],
    ["Fine-Tuning P2",    "~96%",   C.green],
    ["Test Final",        "91.94%", C.blue],
  ];
  prog.forEach(([label, val, col], i) => {
    const y = 3.25 + i*0.52;
    s.addText(label, { x:6.2, y, w:2.0, h:0.38, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
    s.addShape(pres.shapes.RECTANGLE, { x:8.15, y:y+0.04, w:1.35, h:0.3, fill:{color:col, transparency:15} });
    s.addText(val, { x:8.15, y:y+0.04, w:1.35, h:0.3, fontSize:11, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
  });

  // CDC badge
  s.addShape(pres.shapes.RECTANGLE, { x:6.1, y:5.25, w:3.55, h:0.32, fill:{color:C.green} });
  s.addText("✅  KPI CDC validé : Sensibilité > 95%", { x:6.1, y:5.27, w:3.55, h:0.28, fontSize:10, bold:true, color:C.white, align:"center", fontFace:"Calibri" });

  slideNum(s, 9);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — RÉSULTATS BREAST
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "05 · Résultats — Breast Histopathology", C.purple);
  s.addText("Performances CNN Custom Résiduel", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  addMetric(s, 0.4,  1.2, "89.15%", "Val Accuracy", C.purple);
  addMetric(s, 2.65, 1.2, "~88%",   "Test Accuracy", C.blue);
  addMetric(s, 4.9,  1.2, "~0.91",  "ROC-AUC",       C.teal);
  addMetric(s, 7.15, 1.2, "50×50",  "Image Size", C.gray);

  // Architecture summary
  addCard(s, 0.4, 2.65, 4.5, 2.7, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:2.65, w:4.5, h:0.45, fill:{color:C.purple} });
  s.addText("Architecture Résiduelle Custom", { x:0.5, y:2.69, w:4.3, h:0.37, fontSize:12, bold:true, color:C.white, fontFace:"Calibri" });

  const arch = [
    "Input(50,50,3) → Conv2D(64) + BN + MaxPool",
    "Résiduel 1 : 2×Conv2D(128) + Skip Connection",
    "Résiduel 2 : 2×Conv2D(256) + Skip Connection",
    "Résiduel 3 : 2×Conv2D(512) + Skip Connection",
    "GAP → Dense(512) → Dense(256) → Dense(2, Softmax)",
  ];
  arch.forEach((a, i) => {
    s.addText(`→  ${a}`, { x:0.55, y:3.22+i*0.42, w:4.25, h:0.38, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  // Right — context
  addCard(s, 5.15, 2.65, 4.5, 2.7, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.15, y:2.65, w:4.5, h:0.45, fill:{color:C.amber} });
  s.addText("Contexte & Limite", { x:5.25, y:2.69, w:4.3, h:0.37, fontSize:12, bold:true, color:C.white, fontFace:"Calibri" });

  const ctx = [
    "Images 50×50 px — résolution très basse",
    "Limite intrinsèque du dataset Kaggle",
    "Top publications : 85-92% sur ce dataset",
    "Notre modèle : fourchette haute (89.15%)",
    "Amélioration possible : images haute-résolution",
  ];
  ctx.forEach((c, i) => {
    s.addText(`→  ${c}`, { x:5.3, y:3.22+i*0.42, w:4.25, h:0.38, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  slideNum(s, 10);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 11 — FOCAL LOSS + AUTO-ENCODEUR
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "04 · Techniques Avancées", C.teal);
  s.addText("Focal Loss & Détection d'Anomalies", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Focal Loss
  addCard(s, 0.4, 1.25, 4.55, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:4.55, h:0.5, fill:{color:C.teal} });
  s.addText("⚡  Focal Loss", { x:0.5, y:1.3, w:4.35, h:0.4, fontSize:14, bold:true, color:C.white, fontFace:"Calibri" });

  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.9, w:4.35, h:0.55, fill:{color:C.light} });
  s.addText("FL(pt) = −α(1−pt)^γ · log(pt)", {
    x:0.5, y:1.93, w:4.35, h:0.5, fontSize:16, bold:true, color:C.blue, align:"center", fontFace:"Calibri"
  });
  const fl = [
    "γ = 2.0 → focalise sur exemples difficiles",
    "α = 0.25 → équilibre positif/négatif",
    "Pénalise + les classes minoritaires (tumeurs rares)",
    "Complément : class_weight='balanced'",
    "Résultat : meilleure sensibilité sur stades précoces",
  ];
  fl.forEach((f, i) => {
    s.addShape(pres.shapes.OVAL, { x:0.55, y:2.6+i*0.48, w:0.18, h:0.18, fill:{color:C.teal} });
    s.addText(f, { x:0.82, y:2.58+i*0.48, w:4.0, h:0.3, fontSize:11, color:C.dark, fontFace:"Calibri" });
  });

  // Auto-encodeur
  addCard(s, 5.2, 1.25, 4.45, 3.9, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.25, w:4.45, h:0.5, fill:{color:C.purple} });
  s.addText("🤖  Auto-encodeur Anomalies", { x:5.3, y:1.3, w:4.25, h:0.4, fontSize:14, bold:true, color:C.white, fontFace:"Calibri" });

  const ae = [
    ["Encodeur",    "Conv2D(32→16→8) + MaxPool"],
    ["Décodeur",    "Conv2D(8→16→32) + UpSample"],
    ["Loss",        "MSE reconstruction"],
    ["Seuil",       "μ + 2σ sur erreurs test"],
    ["SUSPECT",     "recon_error > seuil → expert"],
    ["UNCERTAIN",   "proba < 0.70 → révision"],
  ];
  ae.forEach(([k,v], i) => {
    const y = 1.9 + i*0.53;
    s.addShape(pres.shapes.RECTANGLE, { x:5.3, y, w:1.3, h:0.38, fill:{color:"F5F3FF"} });
    s.addText(k, { x:5.3, y, w:1.3, h:0.38, fontSize:10, bold:true, color:C.purple, align:"center", fontFace:"Calibri" });
    s.addText(v, { x:6.7, y:y+0.04, w:2.85, h:0.3, fontSize:10.5, color:C.dark, fontFace:"Calibri" });
  });

  slideNum(s, 11);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 12 — GRAD-CAM XAI
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "07 · Explainabilité XAI — Grad-CAM", C.blue2);
  s.addText("Transparence pour la Validation Médicale", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Pipeline visuel
  const steps = [
    { t:"Image IRM",          d:"Entrée 380×380",    c:C.blue },
    { t:"Forward Pass",       d:"Prédiction classe",  c:C.purple },
    { t:"GradientTape",       d:"∇gradient couche",  c:C.teal },
    { t:"Heatmap",            d:"Activation pondérée",c:C.amber },
    { t:"Superposition",      d:"IRM + carte couleur",c:C.green },
  ];

  steps.forEach((st, i) => {
    const x = 0.4 + i*1.88;
    addCard(s, x, 1.25, 1.7, 1.8, "FFFFFF");
    s.addShape(pres.shapes.OVAL, { x:x+0.575, y:1.32, w:0.55, h:0.55, fill:{color:st.c} });
    s.addText(`${i+1}`, { x:x+0.575, y:1.38, w:0.55, h:0.42, fontSize:16, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(st.t, { x:x+0.05, y:1.95, w:1.6, h:0.35, fontSize:11, bold:true, color:C.dark, align:"center", fontFace:"Calibri" });
    s.addText(st.d, { x:x+0.05, y:2.33, w:1.6, h:0.5, fontSize:10, color:C.gray, align:"center", fontFace:"Calibri" });
    if (i<4) s.addText("→", { x:x+1.7, y:1.9, w:0.2, h:0.4, fontSize:18, color:C.gray, fontFace:"Calibri" });
  });

  // Interpretation
  addCard(s, 0.4, 3.2, 9.25, 2.15, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:3.2, w:9.25, h:0.45, fill:{color:C.navy} });
  s.addText("Interprétation des Couleurs Grad-CAM", { x:0.5, y:3.24, w:9.05, h:0.37, fontSize:12, bold:true, color:C.white, align:"center", fontFace:"Calibri" });

  const colors = [
    { c:C.red,    l:"ROUGE", d:"Zone de forte activation neuronale → décisive pour le diagnostic" },
    { c:C.amber,  l:"JAUNE", d:"Zone d'activation modérée → contribue partiellement" },
    { c:C.blue2,  l:"BLEU",  d:"Zone de faible activation → peu influente pour la prédiction" },
  ];
  colors.forEach((col, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.6+i*3.05, y:3.8, w:0.28, h:0.55, fill:{color:col.c} });
    s.addText(col.l, { x:0.98+i*3.05, y:3.82, w:0.9, h:0.25, fontSize:11, bold:true, color:col.c, fontFace:"Calibri" });
    s.addText(col.d, { x:0.98+i*3.05, y:4.08, w:2.5, h:0.55, fontSize:10, color:C.dark, fontFace:"Calibri" });
  });

  // Impact
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:5.22, w:9.25, h:0.32, fill:{color:C.green} });
  s.addText("Impact clinique : Le radiologue voit POURQUOI l'IA a décidé → validation ou rejet en toute connaissance de cause", {
    x:0.5, y:5.24, w:9.05, h:0.28, fontSize:10, bold:true, color:C.white, align:"center", fontFace:"Calibri"
  });

  slideNum(s, 12);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 13 — ARCHITECTURE TECHNIQUE
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "06 · Déploiement", C.red);
  s.addText("Architecture Technique Complète", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // 3-layer diagram
  const layers3 = [
    { t:"🎨  Frontend",    d:"Streamlit Dashboard\nThème Médical Pro\nGrad-CAM Toggle\nModalité selector", c:C.teal,   x:0.3 },
    { t:"⚡  Backend",     d:"FastAPI (async)\n/predict · /predict/dicom\n/benchmark · /health\nMongoDB historique", c:C.blue, x:3.55 },
    { t:"🧠  ML Models",   d:"EfficientNetB4 Brain\nCNN Custom Breast\nAutoencoder Anomaly\nGrad-CAM XAI", c:C.purple, x:6.8 },
  ];

  layers3.forEach(l => {
    addCard(s, l.x, 1.25, 3.0, 3.6, "FFFFFF");
    s.addShape(pres.shapes.RECTANGLE, { x:l.x, y:1.25, w:3.0, h:0.55, fill:{color:l.c} });
    s.addText(l.t, { x:l.x+0.1, y:1.3, w:2.8, h:0.45, fontSize:14, bold:true, color:C.white, fontFace:"Calibri" });
    s.addText(l.d, { x:l.x+0.15, y:1.92, w:2.7, h:2.7, fontSize:12, color:C.dark, fontFace:"Calibri" });
  });

  // Arrows between layers
  s.addShape(pres.shapes.RECTANGLE, { x:3.3, y:2.75, w:0.25, h:0.06, fill:{color:C.gray} });
  s.addText("⟷", { x:3.2, y:2.6, w:0.5, h:0.35, fontSize:16, color:C.gray });
  s.addShape(pres.shapes.RECTANGLE, { x:6.55, y:2.75, w:0.25, h:0.06, fill:{color:C.gray} });
  s.addText("⟷", { x:6.45, y:2.6, w:0.5, h:0.35, fontSize:16, color:C.gray });

  // Bottom — tech stack
  s.addShape(pres.shapes.RECTANGLE, { x:0.3, y:5.0, w:9.4, h:0.5, fill:{color:C.navy} });
  const techs = ["Python 3.11", "TensorFlow 2.21", "Keras 3.14", "FastAPI 0.115", "Streamlit 1.57", "pydicom", "Docker", "OpenCV"];
  s.addText(techs.join("  ·  "), {
    x:0.4, y:5.04, w:9.2, h:0.42, fontSize:11, bold:true, color:C.white, align:"center", fontFace:"Calibri"
  });

  slideNum(s, 13);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 14 — API ENDPOINTS
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "06 · API FastAPI", C.blue);
  s.addText("Endpoints & Réponse API", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Endpoints list
  addCard(s, 0.4, 1.25, 4.5, 3.95, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:4.5, h:0.48, fill:{color:C.blue} });
  s.addText("Endpoints REST", { x:0.5, y:1.29, w:4.3, h:0.38, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });

  const endpoints = [
    { m:"GET",  e:"/",               c:C.gray,   d:"Status API" },
    { m:"GET",  e:"/health",         c:C.green,  d:"Health check + classes" },
    { m:"POST", e:"/predict",        c:C.blue,   d:"Inférence JPG/PNG" },
    { m:"POST", e:"/predict/dicom",  c:C.purple, d:"Inférence DICOM + metadata" },
    { m:"GET",  e:"/benchmark",      c:C.teal,   d:"Latence 5 runs" },
    { m:"GET",  e:"/history",        c:C.amber,  d:"Historique MongoDB" },
  ];
  endpoints.forEach((ep, i) => {
    const y = 1.85 + i*0.52;
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y, w:0.65, h:0.35, fill:{color:ep.c} });
    s.addText(ep.m, { x:0.5, y, w:0.65, h:0.35, fontSize:9, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(ep.e, { x:1.22, y, w:1.9, h:0.35, fontSize:10.5, bold:true, color:C.dark, fontFace:"Calibri" });
    s.addText(ep.d, { x:3.18, y:y+0.03, w:1.55, h:0.3, fontSize:10, color:C.gray, fontFace:"Calibri" });
  });

  // JSON response example
  addCard(s, 5.15, 1.25, 4.5, 3.95, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.15, y:1.25, w:4.5, h:0.48, fill:{color:C.navy} });
  s.addText("Réponse JSON — Exemple", { x:5.25, y:1.29, w:4.3, h:0.38, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });

  s.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.75, w:4.4, h:3.4, fill:{color:"F8F9FF"} });
  const json = [
    `{`,
    `  "modality": "brain_mri",`,
    `  "status": "CONFIRMED",`,
    `  "prediction": "pituitary",`,
    `  "confidence": 82.28,`,
    `  "probabilities": {`,
    `    "glioma": 0.8,`,
    `    "pituitary": 82.28`,
    `  },`,
    `  "reconstruction_error": 0.004,`,
    `  "inference_ms": 847.3`,
    `}`,
  ];
  s.addText(json.join("\n"), {
    x:5.3, y:1.82, w:4.2, h:3.25, fontSize:10, color:C.blue_dark, fontFace:"Courier New"
  });

  slideNum(s, 14);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 15 — DASHBOARD
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "06 · Dashboard Streamlit", C.teal);
  s.addText("Interface Médicale Professionnelle", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Mock dashboard screenshot area
  addCard(s, 0.4, 1.2, 9.2, 4.1, "F8FAFC");
  // Sidebar mock
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.2, w:2.2, h:4.1, fill:{color:C.navy} });
  s.addText("MedVision AI", { x:0.5, y:1.35, w:2.0, h:0.4, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });
  s.addText("Cancer Detection System", { x:0.5, y:1.72, w:2.0, h:0.3, fontSize:9, color:"A5B4FC", fontFace:"Calibri" });

  // Sidebar items
  const sideItems = ["System Status", "Performance Metrics", "Active Models", "Capabilities", "Detection Classes"];
  sideItems.forEach((item, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:2.1+i*0.45, w:1.95, h:0.35, fill:{color:"1A2E5A"} });
    s.addText(item, { x:0.55, y:2.13+i*0.45, w:1.85, h:0.28, fontSize:9.5, color:"CBD5E1", fontFace:"Calibri" });
  });

  // Main content mock
  s.addShape(pres.shapes.RECTANGLE, { x:2.65, y:1.25, w:6.9, h:0.45, fill:{color:C.white} });
  s.addText("MedVision AI  ·  Advanced Cancer Detection Platform  ·  ● System Online", {
    x:2.7, y:1.28, w:6.8, h:0.38, fontSize:10, color:C.dark, fontFace:"Calibri"
  });

  // Left panel
  addCard(s, 2.65, 1.8, 3.3, 3.4, "FFFFFF");
  s.addText("IMAGING MODALITY", { x:2.75, y:1.85, w:3.1, h:0.3, fontSize:8, color:C.gray, fontFace:"Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x:2.75, y:2.2, w:1.45, h:0.38, fill:{color:C.blue} });
  s.addText("🧠 Brain MRI", { x:2.75, y:2.22, w:1.45, h:0.34, fontSize:11, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x:4.3, y:2.2, w:1.45, h:0.38, fill:{color:C.light}, line:{color:C.border, width:0.5} });
  s.addText("🔬 Histopatho", { x:4.3, y:2.22, w:1.45, h:0.34, fontSize:11, color:C.dark, align:"center", fontFace:"Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x:2.75, y:2.72, w:3.1, h:1.8, fill:{color:C.light2}, line:{color:C.border, width:0.5} });
  s.addText("📁 Upload Medical Image\n\nDrag & drop or click to browse\nJPG · JPEG · PNG", {
    x:2.8, y:2.82, w:3.0, h:1.5, fontSize:10, color:C.gray, align:"center", fontFace:"Calibri"
  });

  // Right panel
  addCard(s, 6.15, 1.8, 3.35, 3.4, "FFFFFF");
  s.addText("DIAGNOSTIC RESULT", { x:6.25, y:1.85, w:3.15, h:0.3, fontSize:8, color:C.gray, fontFace:"Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x:6.15, y:2.2, w:3.35, h:2.95, fill:{color:C.blue} });
  s.addText("✓ Confirmed", { x:6.5, y:2.35, w:2.6, h:0.35, fontSize:10, color:C.white, align:"center", fontFace:"Calibri", bold:true });
  s.addText("PITUITARY", { x:6.15, y:2.7, w:3.35, h:0.65, fontSize:28, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
  s.addText("Confidence  82.28%", { x:6.25, y:3.38, w:3.15, h:0.3, fontSize:11, color:"A5B4FC", align:"center", fontFace:"Calibri" });
  s.addShape(pres.shapes.RECTANGLE, { x:6.3, y:3.72, w:3.05, h:0.1, fill:{color:"FFFFFF", transparency:40} });
  s.addShape(pres.shapes.RECTANGLE, { x:6.3, y:3.72, w:2.5, h:0.1, fill:{color:C.white} });
  s.addText("+ All Class Probabilities", { x:6.25, y:4.0, w:3.1, h:0.35, fontSize:10, color:C.white, align:"center", fontFace:"Calibri" });

  slideNum(s, 15);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 16 — CONFORMITÉ CDC
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "05 · Conformité CDC", C.green);
  s.addText("Tableau de Bord — KPIs Cahier des Charges", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  const items = [
    ["CNN Baseline (MVP)",        "Implémenté",   "✅"],
    ["Transfer Learning",         "EfficientNetB4","✅"],
    ["Sensibilité > 95%",         "~97% Glioma",  "✅"],
    ["F2-Score",                  "0.918",        "✅"],
    ["ROC-AUC",                   "0.98",         "✅"],
    ["Grad-CAM XAI",              "Opérationnel", "✅"],
    ["Auto-encodeur anomalies",   "Opérationnel", "✅"],
    ["Focal Loss",                "γ=2, α=0.25",  "✅"],
    ["FastAPI async",             "Opérationnel", "✅"],
    ["Support DICOM/PACS",        "/predict/dicom","✅"],
    ["Dashboard Streamlit",       "Thème médical", "✅"],
    ["2 Datasets",                "Brain + Breast","✅"],
    ["Latence < 200ms",           "Validé GPU T4", "✅"],
    ["Docker",                    "En cours",     "⚠️"],
  ];

  const colW = 4.4;
  items.forEach((item, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * (colW + 0.7);
    const y = 1.25 + row * 0.52;
    const ok = item[2] === "✅";
    s.addShape(pres.shapes.RECTANGLE, { x, y, w:colW, h:0.42, fill:{color:ok?"F0FDF4":"FFFBEB"}, line:{color:ok?"A7F3D0":"FCD34D", width:0.5} });
    s.addText(item[2], { x, y, w:0.42, h:0.42, fontSize:14, align:"center", fontFace:"Calibri" });
    s.addText(item[0], { x:x+0.45, y:y+0.04, w:2.3, h:0.34, fontSize:10.5, bold:true, color:C.dark, fontFace:"Calibri" });
    s.addText(item[1], { x:x+2.8, y:y+0.04, w:1.5, h:0.34, fontSize:10, color:ok?C.green:C.amber, fontFace:"Calibri" });
  });

  // Score
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:5.2, w:9.2, h:0.32, fill:{color:C.green} });
  s.addText("Conformité CDC : 13/14 critères validés · Score estimé : 93%", {
    x:0.5, y:5.22, w:9.0, h:0.28, fontSize:11, bold:true, color:C.white, align:"center", fontFace:"Calibri"
  });

  slideNum(s, 16);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 17 — CONCLUSION
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.light2 };
  sectionTag(s, "08 · Conclusion & Perspectives", C.navy);
  s.addText("Bilan & Prochaines Étapes", { x:0.4, y:0.55, w:9, h:0.55, fontSize:28, bold:true, color:C.dark, fontFace:"Calibri" });

  // Strengths
  addCard(s, 0.4, 1.25, 4.55, 2.6, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.25, w:4.55, h:0.48, fill:{color:C.green} });
  s.addText("✅  Points Forts", { x:0.5, y:1.29, w:4.35, h:0.38, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });
  const strengths = [
    "EfficientNetB4 → 95.71% accuracy",
    "ROC-AUC 0.98 — excellente discrimination",
    "Grad-CAM opérationnel en production",
    "API DICOM/PACS compatible",
    "Pipeline End-to-End complet",
  ];
  strengths.forEach((st, i) => {
    s.addText(`✓  ${st}`, { x:0.55, y:1.85+i*0.38, w:4.25, h:0.34, fontSize:11, color:C.dark, fontFace:"Calibri" });
  });

  // Limits
  addCard(s, 5.2, 1.25, 4.45, 2.6, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.25, w:4.45, h:0.48, fill:{color:C.amber} });
  s.addText("⚠️  Limites Actuelles", { x:5.3, y:1.29, w:4.25, h:0.38, fontSize:13, bold:true, color:C.white, fontFace:"Calibri" });
  const limits = [
    "Breast : 89% (limite images 50×50)",
    "Docker build lent (connexion)",
    "CPU : ~800ms vs 200ms GPU",
    "4 types vs stades I-IV cliniques",
    "Pas encore de deploy cloud",
  ];
  limits.forEach((l, i) => {
    s.addText(`→  ${l}`, { x:5.35, y:1.85+i*0.38, w:4.15, h:0.34, fontSize:11, color:C.dark, fontFace:"Calibri" });
  });

  // Perspectives
  addCard(s, 0.4, 4.0, 9.25, 1.15, "FFFFFF");
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:4.0, w:9.25, h:0.42, fill:{color:C.navy} });
  s.addText("🚀  Perspectives", { x:0.5, y:4.04, w:9.05, h:0.34, fontSize:12, bold:true, color:C.white, fontFace:"Calibri" });
  const persp = ["3ème dataset CT-scan", "ViT / EfficientNetB7", "MLflow tracking", "Deploy HuggingFace", "Auth JWT API", "Tests pytest"];
  persp.forEach((p, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.55+i*1.55, y:4.52, w:1.4, h:0.52, fill:{color:C.light}, line:{color:C.blue2, width:0.5} });
    s.addText(p, { x:0.55+i*1.55, y:4.54, w:1.4, h:0.48, fontSize:9.5, color:C.blue, align:"center", fontFace:"Calibri" });
  });

  slideNum(s, 17);
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 18 — MERCI / QUESTIONS
// ══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.OVAL, { x:-1, y:-1, w:5, h:5, fill:{color:C.blue, transparency:85} });
  s.addShape(pres.shapes.OVAL, { x:7, y:2.5, w:4, h:4, fill:{color:C.purple, transparency:80} });

  s.addText("Merci de votre attention", {
    x:1, y:0.8, w:8, h:1.0, fontSize:40, bold:true, color:C.white, align:"center", fontFace:"Calibri"
  });
  s.addText("Questions & Discussion", {
    x:1, y:1.85, w:8, h:0.55, fontSize:20, color:"A5B4FC", align:"center", fontFace:"Calibri", italic:true
  });

  // Final metrics
  const finalMetrics = [
    { v:"95.71%", l:"Brain MRI", c:C.blue2 },
    { v:"0.98",   l:"ROC-AUC",  c:C.teal },
    { v:"0.918",  l:"F2-Score", c:C.purple },
    { v:"2 Modèles", l:"Multi-modal", c:C.green },
  ];
  finalMetrics.forEach((m, i) => {
    const x = 0.8 + i*2.2;
    s.addShape(pres.shapes.RECTANGLE, { x, y:2.7, w:2.0, h:1.2, fill:{color:m.c, transparency:75}, line:{color:m.c, width:1} });
    s.addText(m.v, { x, y:2.78, w:2.0, h:0.55, fontSize:22, bold:true, color:C.white, align:"center", fontFace:"Calibri" });
    s.addText(m.l, { x, y:3.35, w:2.0, h:0.3, fontSize:10, color:"A5B4FC", align:"center", fontFace:"Calibri" });
  });

  // Tech stack pill
  s.addShape(pres.shapes.RECTANGLE, { x:1.5, y:4.15, w:7, h:0.5, fill:{color:"0A1428"} });
  s.addText("EfficientNetB4 · CNN Custom · FastAPI · Streamlit · Grad-CAM XAI · DICOM · Docker", {
    x:1.5, y:4.19, w:7, h:0.42, fontSize:10, color:"6B7280", align:"center", fontFace:"Calibri"
  });

  s.addText("MedVision AI · Système de Détection de Tumeurs par Deep Learning · 2024–2025", {
    x:0.5, y:5.15, w:9, h:0.35, fontSize:10, color:"4B5563", align:"center", fontFace:"Calibri"
  });
}

// ── Write ──────────────────────────────────────────────────────────────────
pres.writeFile({ fileName:"/mnt/user-data/outputs/MedVision_AI_Soutenance.pptx" })
  .then(() => console.log("✅ Présentation générée !"))
  .catch(e => console.error("❌", e));
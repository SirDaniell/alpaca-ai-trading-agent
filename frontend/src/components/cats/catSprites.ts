/**
 * Every sheet is a 4x4 grid (4 frames per behavior row, 4 rows).
 * Row order is identical across cats so behaviors are interchangeable.
 */
export const SHEET_COLUMNS = 4;
export const SHEET_ROWS = 4;

export type CatVariant = "marmalade" | "smoke" | "mochi";
export type CatBehavior =
  | "idle"
  | "walk"
  | "sit"
  | "lick"
  | "stretch"
  | "pounce"
  | "fight"
  | "knead"
  | "yawn";

type BehaviorSpec = {
  /** Row index in the sprite sheet */
  row: number;
  /** Frames used from that row (left to right) */
  frames: number;
  /** Seconds for one full loop at speed = 1 */
  duration: number;
  /** Does the cat move across the ground while doing this? */
  moves: boolean;
  label: string;
  /** Optional body-movement keyframe layered on top of the sprite frames */
  motion?: "cat-pounce" | "cat-fight" | "cat-knead" | "cat-yawn" | "cat-bob";
  /** Seconds for one loop of that body movement (defaults to duration) */
  motionDuration?: number;
};

export const BEHAVIORS: Record<CatBehavior, BehaviorSpec> = {
  walk: {
    row: 0,
    frames: 4,
    duration: 0.56,
    moves: true,
    label: "Walk",
    motion: "cat-bob",
    motionDuration: 0.28,
  },
  idle: { row: 1, frames: 4, duration: 3.2, moves: false, label: "Idle" },
  sit: { row: 1, frames: 4, duration: 2.2, moves: false, label: "Sit" },
  lick: { row: 2, frames: 4, duration: 1.1, moves: false, label: "Lick paw" },
  stretch: { row: 3, frames: 4, duration: 1.8, moves: false, label: "Stretch" },
  pounce: {
    row: 3,
    frames: 4,
    duration: 1.1,
    moves: false,
    label: "Pounce",
    motion: "cat-pounce",
    motionDuration: 1.1,
  },
  fight: {
    row: 2,
    frames: 4,
    duration: 0.45,
    moves: false,
    label: "Paw throw",
    motion: "cat-fight",
    motionDuration: 0.9,
  },
  knead: {
    row: 2,
    frames: 4,
    duration: 0.6,
    moves: false,
    label: "Knead",
    motion: "cat-knead",
    motionDuration: 0.6,
  },
  yawn: {
    row: 1,
    frames: 4,
    duration: 2.4,
    moves: false,
    label: "Yawn",
    motion: "cat-yawn",
    motionDuration: 2.4,
  },
};

export const BEHAVIOR_LIST = Object.keys(BEHAVIORS) as CatBehavior[];

type VariantSpec = {
  name: string;
  sheet: string;
  /** Multiplies animation speed and walk speed */
  tempo: number;
  /** Pixels per second travelled while walking, at scale 1 */
  walkSpeed: number;
  /** Weights for picking the next autonomous behavior */
  weights: Record<CatBehavior, number>;
  /** [min, max] seconds spent on a non-walking behavior */
  dwell: [number, number];
};

export const CATS: Record<CatVariant, VariantSpec> = {
  marmalade: {
    name: "Marmalade",
    sheet: "/marmalade.png",
    tempo: 1.15,
    walkSpeed: 46,
    weights: {
      walk: 6,
      idle: 2,
      sit: 1.5,
      lick: 1,
      stretch: 1,
      pounce: 1.2,
      fight: 1.5,
      knead: 0.5,
      yawn: 0.5,
    },
    dwell: [1.2, 3],
  },
  smoke: {
    name: "Smoke",
    sheet: "/smoke.png",
    tempo: 0.8,
    walkSpeed: 26,
    weights: {
      walk: 2.5,
      idle: 3,
      sit: 3,
      lick: 1,
      stretch: 3,
      pounce: 0.3,
      fight: 0.3,
      knead: 1,
      yawn: 2.5,
    },
    dwell: [2.5, 6],
  },
  mochi: {
    name: "Mochi",
    sheet: "/mochi.png",
    tempo: 1.05,
    walkSpeed: 38,
    weights: {
      walk: 3.5,
      idle: 1.5,
      sit: 1.5,
      lick: 4,
      stretch: 1.5,
      pounce: 1,
      fight: 1,
      knead: 2.5,
      yawn: 1,
    },
    dwell: [1.5, 4],
  },
};

export const CAT_LIST = Object.keys(CATS) as CatVariant[];

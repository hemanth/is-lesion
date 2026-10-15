export interface BoxCoordinates {
  top: number;
  bottom: number;
  left: number;
  right: number;
  confidence: number | null;
  pixels: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
}

export interface SeverityResult {
  score: number;
  max: number;
  confidence: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
}

export type LesionEngine = 'jev' | 'jev-text' | 'clip-exemplar' | 'clip-zeroshot' | 'ollama' | 'openjev';

export interface IsLesionOptions {
  dataset?: string;
  choices?: string[];
  engine?: LesionEngine;
  visionModel?: string;
  baseUrl?: string;
  useExemplars?: boolean;
  confidenceThreshold?: number;
  apiKey?: string;
  model?: string;
  groundTruth?: string;
  offset?: number;
  row?: number;
  config?: string;
  split?: string;
}

export interface IsLesionResult {
  lesion: boolean;
  tumor: boolean;
  cancer: boolean;
  probability: number;
  noul: number;
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
  severity: SeverityResult;
  box: BoxCoordinates | null;
  triage: 'HIGH_CONFIDENCE_LESION_FINDING' | 'HIGH_CONFIDENCE_ONCOLOGY_FINDING' | 'HIGH_CONFIDENCE_CLEAR_SCAN' | 'ESCALATE_RADIOLOGIST_REVIEW';
  needsReview: boolean;
  dataset: string;
  choices: string[];
  groundTruth: string | null;
  image: {
    width: number;
    height: number;
    format: string;
  };
  model: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  } | null;
  latencyMs: number;
  withDataset: (nextDatasetId: string, nextOpts?: IsLesionOptions) => Promise<IsLesionResult>;
  withEngine: (nextEngine: LesionEngine, nextOpts?: IsLesionOptions) => Promise<IsLesionResult>;
}

declare function isLesion(
  input: string | Buffer | ArrayBuffer | { src: string },
  options?: IsLesionOptions
): Promise<IsLesionResult>;

declare namespace isLesion {
  function fromHF(datasetId?: string, options?: IsLesionOptions): Promise<IsLesionResult & {
    rowIndex: number;
    imageUrl: string;
    groundTruthLesion: boolean | null;
    groundTruthCancer: boolean | null;
    correctChoice: boolean | null;
    correctBinary: boolean | null;
  }>;
  function choices(datasetId?: string, options?: IsLesionOptions): Promise<string[]>;
  function datasets(query?: string, limit?: number): Promise<Array<{
    id: string;
    downloads: number;
    likes: number;
    description: string;
  }>>;
  function engines(): Array<{
    id: LesionEngine;
    label: string;
    runtime: string;
    model: string;
    primitives: string[];
  }>;
}

export default isLesion;
export { isLesion, isLesion as isCancer, IsLesionOptions as IsCancerOptions, IsLesionResult as IsCancerResult };

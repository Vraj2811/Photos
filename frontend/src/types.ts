export interface SystemStatus {
  total_images: number
  total_vectors: number
  ollama_connected: boolean
  models_available: string[]
  status: string
}

export interface SearchResult {
  image_id: number
  filename: string
  description: string
  confidence: number
  image_url: string
}

export interface ImageInfo {
  id: number
  filename: string
  description: string
  created_at: string
  image_url: string
}

export interface UploadResponse {
  success: boolean
  image_id?: number
  filename?: string
  description?: string
  image_url?: string
  error?: string
}






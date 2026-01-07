import axios from 'axios'
import type { SystemStatus, SearchResult, ImageInfo, UploadResponse, FaceGroupInfo } from './types'

const API_BASE = 'http://192.168.1.20:8000'

const api = axios.create({
  baseURL: API_BASE,
})

export const getStatus = async (): Promise<SystemStatus> => {
  const response = await api.get('/api/status')
  return response.data
}

export const searchImages = async (query: string, topK: number = 5): Promise<SearchResult[]> => {
  const response = await api.post('/api/search', { query, top_k: topK })
  return response.data
}

export const getAllImages = async (limit: number = 50, offset: number = 0): Promise<ImageInfo[]> => {
  const response = await api.get('/api/images', { params: { limit, offset } })
  return response.data
}

export const getImage = async (imageId: number): Promise<ImageInfo> => {
  const response = await api.get(`/api/images/${imageId}`)
  return response.data
}

export const uploadImage = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const rebuildIndex = async (): Promise<{ success: boolean; count: number }> => {
  const response = await api.post('/api/rebuild-index')
  return response.data
}

export const deleteImage = async (imageId: number): Promise<{ success: boolean }> => {
  const response = await api.delete(`/api/images/${imageId}`)
  return response.data
}

export const getFaceGroups = async (): Promise<FaceGroupInfo[]> => {
  const response = await api.get('/api/face-groups')
  return response.data
}

export const getFaceGroupImages = async (groupId: number, limit: number = 50, offset: number = 0): Promise<ImageInfo[]> => {
  const response = await api.get(`/api/face-groups/${groupId}`, { params: { limit, offset } })
  return response.data
}
import axios from 'axios'
import type { SystemStatus, SearchResult, ImageInfo, UploadResponse, FaceGroupInfo, Folder } from './types'

const API_BASE = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
})

export const getStatus = async (folderId?: number): Promise<SystemStatus> => {
  const response = await api.get('/api/status', { params: { folder_id: folderId } })
  return response.data
}

export const searchImages = async (query: string, topK: number = 5, folderId?: number): Promise<SearchResult[]> => {
  const response = await api.post('/api/search', { query, top_k: topK, folder_id: folderId })
  return response.data
}

export const getAllImages = async (limit: number = 50, offset: number = 0, folderId?: number): Promise<ImageInfo[]> => {
  const response = await api.get('/api/images', { params: { limit, offset, folder_id: folderId } })
  return response.data
}

export const getImage = async (imageId: number): Promise<ImageInfo> => {
  const response = await api.get(`/api/images/${imageId}`)
  return response.data
}

export const uploadImage = async (file: File, folderId?: number): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/api/upload', formData, {
    params: { folder_id: folderId },
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

export const getFaceGroups = async (folderId?: number): Promise<FaceGroupInfo[]> => {
  const response = await api.get('/api/face-groups', { params: { folder_id: folderId } })
  return response.data
}

export const getFaceGroupImages = async (groupId: number, limit: number = 50, offset: number = 0, folderId?: number): Promise<ImageInfo[]> => {
  const response = await api.get(`/api/face-groups/${groupId}`, { params: { limit, offset, folder_id: folderId } })
  return response.data
}

// Folder API
export const getFolders = async (): Promise<Folder[]> => {
  const response = await api.get('/api/folders')
  return response.data
}

export const createFolder = async (name: string): Promise<Folder> => {
  const response = await api.post('/api/folders', { name })
  return response.data
}

export const deleteFolder = async (folderId: number): Promise<{ success: boolean }> => {
  const response = await api.delete(`/api/folders/${folderId}`)
  return response.data
}
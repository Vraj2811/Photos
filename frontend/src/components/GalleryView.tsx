import { useState, useEffect } from 'react'
import { Loader2, Image as ImageIcon, RefreshCw } from 'lucide-react'
import { getAllImages } from '../api'
import type { ImageInfo } from '../types'
import ImageCard from './ImageCard'

interface GalleryViewProps {
  onImageClick: (image: ImageInfo) => void
  refreshTrigger: number
}

export default function GalleryView({ onImageClick, refreshTrigger }: GalleryViewProps) {
  const [images, setImages] = useState<ImageInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadImages()
  }, [refreshTrigger])

  const loadImages = async () => {
    setLoading(true)
    setError(null)

    try {
      const data = await getAllImages(100)
      setImages(data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load images')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="glass rounded-2xl shadow-2xl p-12 text-center">
        <Loader2 className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
        <p className="text-gray-600">Loading gallery...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass rounded-2xl shadow-2xl p-12 text-center">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={loadImages}
          className="px-6 py-3 bg-indigo-600 text-white rounded-xl font-semibold hover:bg-indigo-700"
        >
          Try Again
        </button>
      </div>
    )
  }

  return (
    <div className="glass rounded-2xl shadow-2xl p-6 md:p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold text-gray-800 mb-2">
            Image Gallery
          </h2>
          <p className="text-gray-600">
            Showing {images.length} image{images.length !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          onClick={loadImages}
          className="p-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Gallery Grid */}
      {images.length === 0 ? (
        <div className="text-center py-12">
          <ImageIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500 text-lg">No images yet</p>
          <p className="text-gray-400 text-sm">Upload some images to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {images.map((image) => (
            <ImageCard
              key={image.id}
              imageUrl={`http://localhost:8000${image.image_url}`}
              description={image.description}
              filename={image.filename}
              onClick={() => onImageClick(image)}
            />
          ))}
        </div>
      )}
    </div>
  )
}






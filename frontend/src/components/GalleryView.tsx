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
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(true)
  const [offset, setOffset] = useState(0)
  const PAGE_SIZE = 30

  useEffect(() => {
    setImages([])
    setOffset(0)
    setHasMore(true)
    loadImages(0, true)
  }, [refreshTrigger])

  const loadImages = async (currentOffset: number, isInitial: boolean = false) => {
    if (isInitial) {
      setLoading(true)
    } else {
      setLoadingMore(true)
    }
    setError(null)

    try {
      const data = await getAllImages(PAGE_SIZE, currentOffset)
      if (data.length < PAGE_SIZE) {
        setHasMore(false)
      }

      if (isInitial) {
        setImages(data)
      } else {
        setImages(prev => [...prev, ...data])
      }
      setOffset(currentOffset + data.length)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load images')
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }

  const handleLoadMore = () => {
    if (!loadingMore && hasMore) {
      loadImages(offset)
    }
  }

  if (loading && images.length === 0) {
    return (
      <div className="glass rounded-2xl shadow-2xl p-12 text-center">
        <Loader2 className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
        <p className="text-gray-600">Loading gallery...</p>
      </div>
    )
  }

  if (error && images.length === 0) {
    return (
      <div className="glass rounded-2xl shadow-2xl p-12 text-center">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => loadImages(0, true)}
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
          onClick={() => {
            setImages([])
            setOffset(0)
            setHasMore(true)
            loadImages(0, true)
          }}
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
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {images.map((image) => (
              <ImageCard
                key={image.id}
                imageUrl={`http://localhost:8000${image.thumbnail_url || image.image_url}`}
                description={image.description}
                filename={image.filename}
                onClick={() => onImageClick(image)}
              />
            ))}
          </div>

          {hasMore && (
            <div className="mt-12 text-center">
              <button
                onClick={handleLoadMore}
                disabled={loadingMore}
                className="px-8 py-3 bg-indigo-600 text-white rounded-xl font-semibold hover:bg-indigo-700 disabled:opacity-50 transition-all flex items-center mx-auto"
              >
                {loadingMore ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Loading...
                  </>
                ) : (
                  'Load More Images'
                )}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}






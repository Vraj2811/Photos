import { useState, useEffect } from 'react'
import { Loader2, Image as ImageIcon, RefreshCw, X, Trash2 } from 'lucide-react'
import { getAllImages, deleteImage } from '../api'
import type { ImageInfo } from '../types'
import ImageCard from './ImageCard'

export default function GalleryView() {
  const [images, setImages] = useState<ImageInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    loadImages()
  }, [])

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

  const handleDelete = async () => {
    if (!selectedImage) return

    if (!confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
      return
    }

    setDeleting(true)
    try {
      await deleteImage(selectedImage.id)
      setImages(images.filter(img => img.id !== selectedImage.id))
      setSelectedImage(null)
    } catch (err: any) {
      alert('Failed to delete image: ' + (err.response?.data?.detail || err.message))
    } finally {
      setDeleting(false)
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
    <>
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
                onClick={() => setSelectedImage(image)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Full Screen Modal */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={() => setSelectedImage(null)}>
          <div className="relative bg-white rounded-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col md:flex-row" onClick={e => e.stopPropagation()}>

            {/* Close Button */}
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-4 right-4 z-10 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>

            {/* Image Section */}
            <div className="w-full md:w-2/3 bg-gray-100 flex items-center justify-center p-4">
              <img
                src={`http://localhost:8000${selectedImage.image_url}`}
                alt={selectedImage.description}
                className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-lg"
              />
            </div>

            {/* Details Section */}
            <div className="w-full md:w-1/3 p-6 md:p-8 flex flex-col overflow-y-auto">
              <h3 className="text-2xl font-bold text-gray-800 mb-4">Image Details</h3>

              <div className="space-y-6 flex-1">
                <div>
                  <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Description</h4>
                  <p className="text-gray-700 leading-relaxed">
                    {selectedImage.description}
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Filename</h4>
                  <p className="text-gray-600 font-mono text-sm break-all">
                    {selectedImage.filename}
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Created At</h4>
                  <p className="text-gray-600">
                    {new Date(selectedImage.created_at).toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="mt-8 pt-6 border-t border-gray-100">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="w-full py-3 px-4 bg-red-50 text-red-600 rounded-xl font-semibold hover:bg-red-100 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {deleting ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Trash2 className="w-5 h-5" />
                  )}
                  {deleting ? 'Deleting...' : 'Delete Image'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}






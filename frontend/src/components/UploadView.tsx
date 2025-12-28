import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, CheckCircle, XCircle, Loader2, Image as ImageIcon } from 'lucide-react'
import { uploadImage } from '../api'

interface UploadViewProps {
  onUploadSuccess: () => void
}

interface UploadedImage {
  file: File
  preview: string
  status: 'pending' | 'uploading' | 'success' | 'error'
  description?: string
  error?: string
}

export default function UploadView({ onUploadSuccess }: UploadViewProps) {
  const [images, setImages] = useState<UploadedImage[]>([])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newImages = acceptedFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      status: 'pending' as const
    }))
    setImages(prev => [...prev, ...newImages])

    // Auto-upload
    newImages.forEach(img => handleUpload(img))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
      'video/*': ['.mp4', '.mov', '.avi', '.mkv']
    },
    multiple: true
  })

  const handleUpload = async (image: UploadedImage) => {
    setImages(prev => prev.map(img =>
      img.preview === image.preview ? { ...img, status: 'uploading' } : img
    ))

    try {
      const result = await uploadImage(image.file)

      if (result.success) {
        setImages(prev => prev.map(img =>
          img.preview === image.preview
            ? { ...img, status: 'success', description: result.description }
            : img
        ))
        onUploadSuccess()
      } else {
        throw new Error(result.error || 'Upload failed')
      }
    } catch (error: any) {
      setImages(prev => prev.map(img =>
        img.preview === image.preview
          ? { ...img, status: 'error', error: error.message }
          : img
      ))
    }
  }

  const clearCompleted = () => {
    setImages(prev => prev.filter(img => img.status !== 'success'))
  }

  return (
    <div className="glass rounded-2xl shadow-2xl p-6 md:p-8">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${isDragActive
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
          }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-4">
          <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-6 rounded-full shadow-xl">
            <Upload className="w-12 h-12 text-white" />
          </div>
          <div>
            <p className="text-xl font-semibold text-gray-800 mb-2">
              {isDragActive ? 'Drop files here' : 'Upload Images or Videos'}
            </p>
            <p className="text-gray-600">
              Drag & drop files here, or click to select
            </p>
            <p className="text-sm text-gray-400 mt-2">
              Supports: JPG, PNG, GIF, WEBP, MP4, MOV
            </p>
          </div>
        </div>
      </div>

      {/* Upload Queue */}
      {images.length > 0 && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800">
              Upload Queue ({images.length})
            </h3>
            <button
              onClick={clearCompleted}
              className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
            >
              Clear Completed
            </button>
          </div>

          <div className="space-y-4">
            {images.map((image, idx) => (
              <div
                key={idx}
                className="bg-white rounded-xl p-4 shadow-md flex items-center gap-4"
              >
                {/* Thumbnail */}
                <div className="w-24 h-24 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0 flex items-center justify-center">
                  {image.file.type.startsWith('video/') ? (
                    <div className="text-gray-400 flex flex-col items-center">
                      <span className="text-xs font-bold uppercase">Video</span>
                    </div>
                  ) : (
                    <img
                      src={image.preview}
                      alt={image.file.name}
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-gray-800 truncate">
                    {image.file.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {(image.file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  {image.description && (
                    <p className="text-sm text-gray-600 mt-2 line-clamp-2">
                      {image.description}
                    </p>
                  )}
                  {image.error && (
                    <p className="text-sm text-red-600 mt-2">
                      {image.error}
                    </p>
                  )}
                </div>

                {/* Status */}
                <div className="flex-shrink-0">
                  {image.status === 'pending' && (
                    <ImageIcon className="w-8 h-8 text-gray-400" />
                  )}
                  {image.status === 'uploading' && (
                    <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
                  )}
                  {image.status === 'success' && (
                    <CheckCircle className="w-8 h-8 text-green-500" />
                  )}
                  {image.status === 'error' && (
                    <XCircle className="w-8 h-8 text-red-500" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="mt-8 bg-indigo-50 border-l-4 border-indigo-500 rounded-lg p-4">
        <p className="font-semibold text-indigo-900 mb-2">✨ How it works:</p>
        <ol className="text-sm text-indigo-800 space-y-1 list-decimal list-inside">
          <li>Upload your images</li>
          <li>AI (LLaVA) generates a description</li>
          <li>Description is converted to vector embedding</li>
          <li>Stored in database and indexed for search</li>
          <li>Ready to search instantly!</li>
        </ol>
      </div>
    </div>
  )
}






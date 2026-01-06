import { useState } from 'react'
import { ImageOff, Loader2 } from 'lucide-react'

interface ImageCardProps {
  imageUrl: string
  description: string
  confidence?: number
  filename: string
  isProcessing?: boolean
  onClick?: () => void
}

export default function ImageCard({ imageUrl, description, confidence, filename, isProcessing, onClick }: ImageCardProps) {
  const [error, setError] = useState(false)

  const getConfidenceBadge = (score: number) => {
    if (score >= 0.8) return { label: 'Excellent', color: 'bg-green-500' }
    if (score >= 0.6) return { label: 'Good', color: 'bg-blue-500' }
    if (score >= 0.4) return { label: 'Fair', color: 'bg-yellow-500' }
    return { label: 'Low', color: 'bg-gray-400' }
  }

  const badge = confidence !== undefined ? getConfidenceBadge(confidence) : null

  return (
    <div
      onClick={onClick}
      className="group bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 hover:scale-105 cursor-pointer"
    >
      {/* Image */}
      <div className="relative aspect-video overflow-hidden bg-gray-100 flex items-center justify-center">
        {error ? (
          <div className="flex flex-col items-center text-gray-400 p-4">
            <ImageOff className="w-8 h-8 mb-2" />
            <span className="text-xs text-center">Image unavailable</span>
          </div>
        ) : (
          <img
            src={imageUrl}
            alt={description}
            onError={() => setError(true)}
            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
            loading="lazy"
          />
        )}
        {badge && (
          <div className={`absolute top-3 right-3 ${badge.color} text-white px-3 py-1 rounded-full text-xs font-semibold shadow-lg`}>
            {badge.label} {(confidence! * 100).toFixed(0)}%
          </div>
        )}
        {isProcessing && (
          <div className="absolute inset-0 bg-black/20 backdrop-blur-[1px] flex items-center justify-center">
            <div className="bg-white/90 px-3 py-1.5 rounded-full flex items-center gap-2 shadow-lg">
              <Loader2 className="w-3 h-3 animate-spin text-indigo-600" />
              <span className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Processing</span>
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        <p className="text-gray-800 text-sm leading-relaxed mb-2 line-clamp-2">
          {description}
        </p>
        <p className="text-xs text-gray-400 truncate">
          {filename}
        </p>
      </div>
    </div>
  )
}






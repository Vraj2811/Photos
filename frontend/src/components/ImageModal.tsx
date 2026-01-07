import { X, Trash2, Loader2 } from 'lucide-react'
import type { ImageInfo } from '../types'

interface ImageModalProps {
    image: ImageInfo
    onClose: () => void
    onDelete: (imageId: number) => Promise<void>
    deleting: boolean
}

export default function ImageModal({ image, onClose, onDelete, deleting }: ImageModalProps) {
    return (
        <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="relative bg-white rounded-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col md:flex-row shadow-2xl"
                onClick={e => e.stopPropagation()}
            >
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
                >
                    <X className="w-6 h-6" />
                </button>

                {/* Image Section */}
                <div className="w-full md:w-2/3 bg-gray-100 flex items-center justify-center p-4 min-h-[300px]">
                    <img
                        src={`http://192.168.1.20:8000${image.image_url}`}
                        alt={image.description}
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
                                {image.description}
                            </p>
                        </div>

                        <div>
                            <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Filename</h4>
                            <p className="text-gray-600 font-mono text-sm break-all">
                                {image.filename}
                            </p>
                        </div>

                        <div>
                            <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Created At</h4>
                            <p className="text-gray-600">
                                {new Date(image.created_at).toLocaleString()}
                            </p>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="mt-8 pt-6 border-t border-gray-100">
                        <button
                            onClick={() => onDelete(image.id)}
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
    )
}

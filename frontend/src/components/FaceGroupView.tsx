import { useState, useEffect } from 'react'
import { ArrowLeft, Activity, Image as ImageIcon } from 'lucide-react'
import { getFaceGroupImages } from '../api'
import type { ImageInfo } from '../types'
import ImageCard from './ImageCard'

interface FaceGroupViewProps {
    groupId: number
    onBack: () => void
    onImageClick: (image: ImageInfo) => void
    refreshTrigger: number
}

export default function FaceGroupView({ groupId, onBack, onImageClick, refreshTrigger }: FaceGroupViewProps) {
    const [images, setImages] = useState<ImageInfo[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadImages()
    }, [groupId, refreshTrigger])

    const loadImages = async () => {
        try {
            const data = await getFaceGroupImages(groupId)
            setImages(data)
        } catch (error) {
            console.error('Failed to load face group images:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="glass rounded-2xl shadow-2xl p-12 text-center">
                <Activity className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
                <p className="text-gray-600">Loading photos...</p>
            </div>
        )
    }

    return (
        <div className="glass rounded-2xl shadow-2xl p-6 md:p-8">
            <div className="space-y-6">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                    >
                        <ArrowLeft className="w-6 h-6 text-gray-600" />
                    </button>
                    <h2 className="text-2xl font-bold text-gray-800">Photos of Person {groupId}</h2>
                </div>

                {images.length === 0 ? (
                    <div className="text-center py-12">
                        <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-600">No photos found for this person.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
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
        </div>
    )
}

import { useState, useEffect } from 'react'
import { ArrowLeft, Activity, Image as ImageIcon, Loader2 } from 'lucide-react'
import { getFaceGroupImages } from '../api'
import type { ImageInfo } from '../types'
import ImageCard from './ImageCard'

interface FaceGroupViewProps {
    groupId: number
    onBack: () => void
    onImageClick: (image: ImageInfo) => void
    refreshTrigger: number
    folderId?: number
}

export default function FaceGroupView({ groupId, onBack, onImageClick, refreshTrigger, folderId }: FaceGroupViewProps) {
    const [images, setImages] = useState<ImageInfo[]>([])
    const [loading, setLoading] = useState(true)
    const [loadingMore, setLoadingMore] = useState(false)
    const [hasMore, setHasMore] = useState(true)
    const [offset, setOffset] = useState(0)
    const PAGE_SIZE = 30

    useEffect(() => {
        setImages([])
        setOffset(0)
        setHasMore(true)
        loadImages(0, true)
    }, [groupId, refreshTrigger, folderId])

    const loadImages = async (currentOffset: number, isInitial: boolean = false) => {
        if (isInitial) {
            setLoading(true)
        } else {
            setLoadingMore(true)
        }

        try {
            const data = await getFaceGroupImages(groupId, PAGE_SIZE, currentOffset, folderId)
            if (data.length < PAGE_SIZE) {
                setHasMore(false)
            }

            if (isInitial) {
                setImages(data)
            } else {
                setImages(prev => [...prev, ...data])
            }
            setOffset(currentOffset + data.length)
        } catch (error) {
            console.error('Failed to load face group images:', error)
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
                    <>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
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
                                        'Load More Photos'
                                    )}
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}

import { useState, useEffect } from 'react'
import { Users, Activity } from 'lucide-react'
import { getFaceGroups } from '../api'
import type { FaceGroupInfo } from '../types'

interface PeopleViewProps {
    onSelectGroup: (groupId: number) => void
}

export default function PeopleView({ onSelectGroup }: PeopleViewProps) {
    const [groups, setGroups] = useState<FaceGroupInfo[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadGroups()
    }, [])

    const loadGroups = async () => {
        try {
            const data = await getFaceGroups()
            setGroups(data)
        } catch (error) {
            console.error('Failed to load face groups:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="glass rounded-2xl shadow-2xl p-12 text-center">
                <Activity className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
                <p className="text-gray-600">Loading people...</p>
            </div>
        )
    }

    if (groups.length === 0) {
        return (
            <div className="glass rounded-2xl shadow-2xl p-12 text-center">
                <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No people detected yet.</p>
                <p className="text-sm text-gray-500 mt-2">Upload images with faces to see them here.</p>
            </div>
        )
    }

    return (
        <div className="glass rounded-2xl shadow-2xl p-6 md:p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-8">People</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                {groups.map((group) => (
                    <div
                        key={group.id}
                        onClick={() => onSelectGroup(group.id)}
                        className="group cursor-pointer"
                    >
                        <div className="relative aspect-square rounded-full overflow-hidden border-4 border-white shadow-lg group-hover:shadow-2xl group-hover:scale-105 transition-all duration-300">
                            {group.representative_image_url ? (
                                <img
                                    src={`http://192.168.1.20:8000${group.representative_image_url}`}
                                    alt={group.name || `Person ${group.id}`}
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <div className="w-full h-full bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center">
                                    <Users className="w-12 h-12 text-indigo-300" />
                                </div>
                            )}
                            <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors duration-300" />
                        </div>
                        <div className="mt-4 text-center">
                            <h3 className="font-semibold text-gray-800 group-hover:text-indigo-600 transition-colors">
                                {group.name || `Person ${group.id}`}
                            </h3>
                            <p className="text-sm text-gray-500">{group.image_count} photos</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

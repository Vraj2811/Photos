import { useState, useEffect } from 'react'
import { Search, Upload, Image as ImageIcon, Activity, Sparkles, Users } from 'lucide-react'
import SearchView from './components/SearchView'
import UploadView from './components/UploadView'
import GalleryView from './components/GalleryView'
import PeopleView from './components/PeopleView'
import FaceGroupView from './components/FaceGroupView'
import ImageModal from './components/ImageModal'
import StatusBar from './components/StatusBar'
import { getStatus, deleteImage as apiDeleteImage } from './api'
import type { SystemStatus, ImageInfo } from './types'

type View = 'search' | 'upload' | 'gallery' | 'people' | 'face-group'

function App() {
  const [currentView, setCurrentView] = useState<View>('search')
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null)
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStatus()
    const interval = setInterval(loadStatus, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [])

  const loadStatus = async () => {
    try {
      const data = await getStatus()
      setStatus(data)
    } catch (error) {
      console.error('Failed to load status:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectGroup = (groupId: number) => {
    setSelectedGroupId(groupId)
    setCurrentView('face-group')
  }

  const handleDeleteImage = async (imageId: number) => {
    if (!confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
      return
    }

    setDeleting(true)
    try {
      await apiDeleteImage(imageId)
      setSelectedImage(null)
      setRefreshTrigger(prev => prev + 1)
      loadStatus() // Refresh status to update counts
      // Note: Individual views will need to refresh their own data if they are active
      // We can use a simple event or just let them refresh on mount
    } catch (err: any) {
      alert('Failed to delete image: ' + (err.response?.data?.detail || err.message))
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="glass rounded-2xl shadow-2xl p-6 mb-6 fade-in">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-3 rounded-xl shadow-lg">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  AI Image Search
                </h1>
                <p className="text-gray-600 text-sm">Powered by LLaVA & Embeddings</p>
              </div>
            </div>

            {status && (
              <div className="hidden md:block">
                <StatusBar status={status} />
              </div>
            )}
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav className="glass rounded-2xl shadow-xl p-2 mb-6 fade-in">
          <div className="flex gap-2">
            <TabButton
              active={currentView === 'search'}
              onClick={() => setCurrentView('search')}
              icon={<Search className="w-5 h-5" />}
              label="Search"
            />
            <TabButton
              active={currentView === 'upload'}
              onClick={() => setCurrentView('upload')}
              icon={<Upload className="w-5 h-5" />}
              label="Upload"
            />
            <TabButton
              active={currentView === 'gallery'}
              onClick={() => setCurrentView('gallery')}
              icon={<ImageIcon className="w-5 h-5" />}
              label="Gallery"
            />
            <TabButton
              active={currentView === 'people' || currentView === 'face-group'}
              onClick={() => setCurrentView('people')}
              icon={<Users className="w-5 h-5" />}
              label="People"
            />
          </div>
        </nav>

        {/* Mobile Status */}
        {status && (
          <div className="md:hidden mb-6">
            <StatusBar status={status} />
          </div>
        )}

        {/* Main Content */}
        <main className="fade-in">
          {loading ? (
            <div className="glass rounded-2xl shadow-2xl p-12 text-center">
              <Activity className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
              <p className="text-gray-600">Loading...</p>
            </div>
          ) : (
            <>
              {currentView === 'search' && (
                <SearchView
                  onImageClick={setSelectedImage}
                  refreshTrigger={refreshTrigger}
                />
              )}
              {currentView === 'upload' && <UploadView onUploadSuccess={loadStatus} />}
              {currentView === 'gallery' && (
                <GalleryView
                  onImageClick={setSelectedImage}
                  refreshTrigger={refreshTrigger}
                />
              )}
              {currentView === 'people' && <PeopleView onSelectGroup={handleSelectGroup} />}
              {currentView === 'face-group' && selectedGroupId && (
                <FaceGroupView
                  groupId={selectedGroupId}
                  onBack={() => setCurrentView('people')}
                  onImageClick={setSelectedImage}
                  refreshTrigger={refreshTrigger}
                />
              )}
            </>
          )}
        </main>

        {/* Global Image Modal */}
        {selectedImage && (
          <ImageModal
            image={selectedImage}
            onClose={() => setSelectedImage(null)}
            onDelete={handleDeleteImage}
            deleting={deleting}
          />
        )}

        {/* Footer */}
        <footer className="mt-8 text-center text-white text-sm opacity-75">
          <p>Built with React, FastAPI, LLaVA & FAISS</p>
        </footer>
      </div>
    </div>
  )
}

interface TabButtonProps {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${active
        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg scale-105'
        : 'text-gray-600 hover:bg-gray-100'
        }`}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  )
}

export default App

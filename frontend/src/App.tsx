import { useState, useEffect } from 'react'
import { Search, Upload, Image as ImageIcon, Activity, Sparkles, Users, Loader2, CheckCircle, XCircle } from 'lucide-react'
import SearchView from './components/SearchView'
import UploadView from './components/UploadView'
import GalleryView from './components/GalleryView'
import PeopleView from './components/PeopleView'
import FaceGroupView from './components/FaceGroupView'
import ImageModal from './components/ImageModal'
import StatusBar from './components/StatusBar'
import { getStatus, deleteImage as apiDeleteImage, getFolders, createFolder } from './api'
import type { SystemStatus, ImageInfo, Folder } from './types'
import { Folder as FolderIcon, Plus } from 'lucide-react'
import { UploadProvider, useUploads } from './UploadContext'

type View = 'search' | 'upload' | 'gallery' | 'people' | 'face-group'

function GlobalUploadProgress() {
  const { uploads, isUploading, clearCompleted } = useUploads();
  const [isExpanded, setIsExpanded] = useState(false);

  if (uploads.length === 0) return null;

  const successCount = uploads.filter(u => u.status === 'success').length;
  const totalCount = uploads.length;

  return (
    <div className={`fixed bottom-6 right-6 z-50 transition-all duration-300 ${isExpanded ? 'w-80' : 'w-64'}`}>
      <div className="glass rounded-2xl shadow-2xl overflow-hidden border border-indigo-100">
        <div
          className="bg-gradient-to-r from-indigo-600 to-purple-600 p-3 flex items-center justify-between cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center gap-2 text-white">
            {isUploading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <CheckCircle className="w-4 h-4" />
            )}
            <span className="text-sm font-medium">
              {isUploading ? 'Uploading...' : 'Uploads Complete'}
            </span>
          </div>
          <span className="text-xs text-white opacity-80">
            {successCount}/{totalCount}
          </span>
        </div>

        {isExpanded && (
          <div className="max-h-64 overflow-y-auto p-2 space-y-2 bg-white/80 backdrop-blur-sm">
            {uploads.map((upload, idx) => (
              <div key={idx} className="flex items-center gap-2 p-2 rounded-lg bg-white/50 text-xs">
                <img src={upload.preview} className="w-8 h-8 rounded object-cover" alt="" />
                <div className="flex-1 min-w-0">
                  <p className="truncate font-medium text-gray-700">{upload.file.name}</p>
                  <p className={`text-[10px] ${upload.status === 'success' ? 'text-green-600' :
                    upload.status === 'error' ? 'text-red-600' : 'text-indigo-600'
                    }`}>
                    {upload.status.charAt(0).toUpperCase() + upload.status.slice(1)}
                  </p>
                </div>
                {upload.status === 'uploading' && <Loader2 className="w-3 h-3 animate-spin text-indigo-600" />}
                {upload.status === 'success' && <CheckCircle className="w-3 h-3 text-green-500" />}
                {upload.status === 'error' && <XCircle className="w-3 h-3 text-red-500" />}
              </div>
            ))}
            {!isUploading && (
              <button
                onClick={(e) => { e.stopPropagation(); clearCompleted(); }}
                className="w-full py-2 text-xs text-indigo-600 hover:text-indigo-800 font-medium border-t border-gray-100"
              >
                Clear Completed
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AppContent() {
  const [currentView, setCurrentView] = useState<View>('search')
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null)
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [folders, setFolders] = useState<Folder[]>([])
  const [selectedFolderId, setSelectedFolderId] = useState<number | undefined>(undefined)
  const [isCreatingFolder, setIsCreatingFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')

  useEffect(() => {
    loadStatus()
    loadFolders()
    const interval = setInterval(loadStatus, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [selectedFolderId])

  const loadStatus = async () => {
    try {
      const data = await getStatus(selectedFolderId)
      setStatus(data)
    } catch (error) {
      console.error('Failed to load status:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadFolders = async () => {
    try {
      const data = await getFolders()
      setFolders(data)
    } catch (error) {
      console.error('Failed to load folders:', error)
    }
  }

  const handleCreateFolder = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newFolderName.trim()) return
    try {
      const newFolder = await createFolder(newFolderName.trim())
      setFolders([...folders, newFolder])
      setNewFolderName('')
      setIsCreatingFolder(false)
      setSelectedFolderId(newFolder.id)
    } catch (error) {
      alert('Failed to create folder')
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
              <div className="flex flex-col md:flex-row items-end md:items-center gap-4">
                {/* Folder Selector */}
                <div className="flex items-center gap-2">
                  <div className="relative group">
                    <select
                      value={selectedFolderId || ''}
                      onChange={(e) => setSelectedFolderId(e.target.value ? Number(e.target.value) : undefined)}
                      className="appearance-none bg-white/50 border border-indigo-100 rounded-xl px-4 py-2 pr-10 text-sm font-medium text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all cursor-pointer hover:bg-white"
                    >
                      <option value="">All Photos</option>
                      {folders.map(folder => (
                        <option key={folder.id} value={folder.id}>{folder.name}</option>
                      ))}
                    </select>
                    <FolderIcon className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-indigo-500 pointer-events-none" />
                  </div>

                  <button
                    onClick={() => setIsCreatingFolder(true)}
                    className="p-2 rounded-xl bg-indigo-50 text-indigo-600 hover:bg-indigo-100 transition-colors"
                    title="New Folder"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                </div>

                <div className="hidden md:block">
                  <StatusBar status={status} />
                </div>
              </div>
            )}
          </div>

          {/* Create Folder Modal Overlay */}
          {isCreatingFolder && (
            <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/20 backdrop-blur-sm">
              <div className="glass rounded-2xl shadow-2xl p-6 w-full max-w-md animate-in zoom-in duration-200">
                <h3 className="text-xl font-bold text-gray-800 mb-4">Create New Folder</h3>
                <form onSubmit={handleCreateFolder}>
                  <input
                    autoFocus
                    type="text"
                    value={newFolderName}
                    onChange={(e) => setNewFolderName(e.target.value)}
                    placeholder="Folder name..."
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4"
                  />
                  <div className="flex gap-3">
                    <button
                      type="button"
                      onClick={() => setIsCreatingFolder(false)}
                      className="flex-1 px-4 py-2 rounded-xl border border-gray-200 text-gray-600 hover:bg-gray-50 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={!newFolderName.trim()}
                      className="flex-1 px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-50"
                    >
                      Create
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
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
                  folderId={selectedFolderId}
                />
              )}
              {currentView === 'upload' && <UploadView folderId={selectedFolderId} />}
              {currentView === 'gallery' && (
                <GalleryView
                  onImageClick={setSelectedImage}
                  refreshTrigger={refreshTrigger}
                  folderId={selectedFolderId}
                />
              )}
              {currentView === 'people' && <PeopleView onSelectGroup={handleSelectGroup} folderId={selectedFolderId} />}
              {currentView === 'face-group' && selectedGroupId && (
                <FaceGroupView
                  groupId={selectedGroupId}
                  onBack={() => setCurrentView('people')}
                  onImageClick={setSelectedImage}
                  refreshTrigger={refreshTrigger}
                  folderId={selectedFolderId}
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

      <GlobalUploadProgress />
    </div>
  )
}

function App() {
  return (
    <UploadProvider>
      <AppContent />
    </UploadProvider>
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

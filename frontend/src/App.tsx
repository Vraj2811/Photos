import { useState, useEffect } from 'react'
import { Search, Upload, Image as ImageIcon, Activity, Sparkles } from 'lucide-react'
import SearchView from './components/SearchView'
import UploadView from './components/UploadView'
import GalleryView from './components/GalleryView'
import StatusBar from './components/StatusBar'
import { getStatus } from './api'
import type { SystemStatus } from './types'

type View = 'search' | 'upload' | 'gallery'

function App() {
  const [currentView, setCurrentView] = useState<View>('search')
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
              {currentView === 'search' && <SearchView />}
              {currentView === 'upload' && <UploadView onUploadSuccess={loadStatus} />}
              {currentView === 'gallery' && <GalleryView />}
            </>
          )}
        </main>

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
      className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
        active
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






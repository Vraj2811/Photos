import { useState, useEffect } from 'react'
import { Search, Loader2, AlertCircle } from 'lucide-react'
import { searchImages } from '../api'
import type { SearchResult, ImageInfo } from '../types'
import ImageCard from './ImageCard'

interface SearchViewProps {
  onImageClick: (image: ImageInfo) => void
  refreshTrigger: number
  folderId?: number
}

export default function SearchView({ onImageClick, refreshTrigger, folderId }: SearchViewProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searched, setSearched] = useState(false)

  useEffect(() => {
    if (searched && query.trim()) {
      handleSearch()
    }
  }, [refreshTrigger, folderId])

  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const data = await searchImages(query, 10, folderId)
      setResults(data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Search failed. Please try again.')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <div className="glass rounded-2xl shadow-2xl p-6 md:p-8">
      {/* Search Input */}
      <div className="mb-8">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Search for images... (e.g., 'Nike shoes', 'sunset', 'person walking')"
              className="w-full px-6 py-4 rounded-xl border-2 border-gray-200 focus:border-indigo-500 focus:outline-none text-lg transition-all"
            />
            <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-6 h-6 text-gray-400" />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
          >
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : (
              <span>Search</span>
            )}
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-2 ml-2">
          💡 Try: "Nike shoes", "motorcycle", "person standing", "sunset with mountains"
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-800">Search Error</p>
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <Loader2 className="w-12 h-12 text-indigo-600 mx-auto mb-4 animate-spin" />
          <p className="text-gray-600">Searching images...</p>
        </div>
      )}

      {/* Results */}
      {!loading && searched && results.length === 0 && !error && (
        <div className="text-center py-12">
          <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500 text-lg">No results found</p>
          <p className="text-gray-400 text-sm">Try different keywords</p>
        </div>
      )}

      {!loading && results.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800">
              Found {results.length} result{results.length !== 1 ? 's' : ''}
            </h2>
            <div className="text-sm text-gray-500">
              for "{query}"
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((result) => (
              <ImageCard
                key={result.image_id}
                imageUrl={`http://localhost:8000${result.thumbnail_url || result.image_url}`}
                description={result.description}
                confidence={result.confidence}
                filename={result.filename}
                onClick={() => onImageClick({
                  id: result.image_id,
                  filename: result.filename,
                  description: result.description,
                  image_url: result.image_url,
                  thumbnail_url: result.thumbnail_url,
                  created_at: new Date().toISOString() // Placeholder since SearchResult doesn't have it
                })}
              />
            ))}
          </div>
        </div>
      )}

      {/* Welcome Message */}
      {!searched && !loading && (
        <div className="text-center py-12">
          <div className="bg-gradient-to-br from-indigo-500 to-purple-600 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl">
            <Search className="w-10 h-10 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-3">
            Search Your Images with AI
          </h2>
          <p className="text-gray-600 max-w-md mx-auto">
            Use natural language to find exactly what you're looking for.
            Our AI understands context and meaning, not just keywords.
          </p>
        </div>
      )}
    </div>
  )
}






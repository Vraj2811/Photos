import React, { createContext, useContext, useState, useCallback } from 'react';
import { uploadImage } from './api';

export interface UploadedImage {
    file: File;
    preview: string;
    status: 'pending' | 'uploading' | 'success' | 'error';
    description?: string;
    error?: string;
    folderId?: number;
}

interface UploadContextType {
    uploads: UploadedImage[];
    addUploads: (files: File[], folderId?: number) => void;
    clearCompleted: () => void;
    isUploading: boolean;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export function UploadProvider({ children, onUploadSuccess }: { children: React.ReactNode, onUploadSuccess?: () => void }) {
    const [uploads, setUploads] = useState<UploadedImage[]>([]);

    const handleUpload = useCallback(async (image: UploadedImage) => {
        setUploads(prev => prev.map(img =>
            img.preview === image.preview ? { ...img, status: 'uploading' } : img
        ));

        try {
            const result = await uploadImage(image.file, image.folderId);

            if (result.success) {
                setUploads(prev => prev.map(img =>
                    img.preview === image.preview
                        ? { ...img, status: 'success', description: result.description }
                        : img
                ));
                if (onUploadSuccess) onUploadSuccess();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error: any) {
            setUploads(prev => prev.map(img =>
                img.preview === image.preview
                    ? { ...img, status: 'error', error: error.message }
                    : img
            ));
        }
    }, [onUploadSuccess]);

    const addUploads = useCallback((files: File[], folderId?: number) => {
        const newUploads = files.map(file => ({
            file,
            preview: URL.createObjectURL(file),
            status: 'pending' as const,
            folderId
        }));

        setUploads(prev => [...prev, ...newUploads]);

        // Start uploads
        newUploads.forEach(img => handleUpload(img));
    }, [handleUpload]);

    const clearCompleted = useCallback(() => {
        setUploads(prev => prev.filter(img => img.status !== 'success'));
    }, []);

    const isUploading = uploads.some(img => img.status === 'uploading' || img.status === 'pending');

    return (
        <UploadContext.Provider value={{ uploads, addUploads, clearCompleted, isUploading }}>
            {children}
        </UploadContext.Provider>
    );
}

export function useUploads() {
    const context = useContext(UploadContext);
    if (context === undefined) {
        throw new Error('useUploads must be used within an UploadProvider');
    }
    return context;
}

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi } from '../api/client';
import { FileText, Send } from 'lucide-react';

export default function JDRetryForm({ jobId }) {
  const [text, setText] = useState('');
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (jdText) => jobsApi.retryAnalysis(jobId, jdText),
    onSuccess: () => {
      // Invalidate the job query to immediately fetch the new 'PENDING' state
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
    },
    onError: (err) => {
      console.error('Failed to retry analysis:', err);
    }
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    mutation.mutate(text);
  };

  if (mutation.isSuccess) {
    return (
      <div className="text-sm text-green-600 font-semibold animate-pulse">
        Analysis restarting with text...
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <textarea
        className="w-full h-32 p-3 text-sm rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary focus:border-primary outline-none transition-all resize-none font-mono"
        placeholder="Paste the raw job description here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={mutation.isPending}
        required
      />
      <button
        type="submit"
        className="btn btn-primary self-end"
        disabled={mutation.isPending || !text.trim()}
      >
        {mutation.isPending ? 'Submitting...' : <><Send className="w-4 h-4" /> Retry Analysis</>}
      </button>
      {mutation.isError && (
        <p className="text-xs text-red-500 mt-1">Failed to submit text. Please try again.</p>
      )}
    </form>
  );
}

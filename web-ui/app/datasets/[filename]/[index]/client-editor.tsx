"use client"

import { useState } from "react";
import { useRouter } from "next/navigation";
import { DatasetEntry, Message, ThinkingBlock } from "@/lib/types";
import { parseThinking, constructThinking } from "@/lib/utils/thinking-parser";
import { saveDatasetEntry } from "@/lib/actions/datasets";
import { ThinkingEditor } from "@/components/thinking-editor";
import { ToolEditor } from "@/components/tool-editor";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { ArrowLeft, Save } from "lucide-react";

interface ClientEditorProps {
  entry: DatasetEntry;
  filename: string;
  index: number;
}

export function ClientEditor({ entry: initialEntry, filename, index }: ClientEditorProps) {
  const router = useRouter();
  const [entry, setEntry] = useState<DatasetEntry>(initialEntry);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await saveDatasetEntry(filename, index, entry);
      toast.success("Entry saved successfully");
    } catch (e) {
      toast.error("Failed to save entry");
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  };

  const updateMessage = (msgIndex: number, newMsg: Message) => {
    const newConversations = [...entry.conversations];
    newConversations[msgIndex] = newMsg;
    setEntry({ ...entry, conversations: newConversations });
  };

  const handleThinkingChange = (msgIndex: number, newThinking: ThinkingBlock) => {
    const msg = entry.conversations[msgIndex];
    const newContent = constructThinking(newThinking);
    updateMessage(msgIndex, { ...msg, content: newContent });
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto pb-20">
      <div className="flex items-center justify-between sticky top-0 bg-background z-10 py-4 border-b">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => router.back()}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <h1 className="text-xl font-bold">Editor: Entry #{index}</h1>
        </div>
        <Button onClick={handleSave} disabled={isSaving}>
          <Save className="mr-2 h-4 w-4" />
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </div>

      <div className="space-y-8">
        {entry.conversations.map((msg, i) => {
          const thinking = parseThinking(msg.content);
          
          return (
            <div key={i} className="space-y-4 border rounded-lg p-6 bg-card">
              <div className="font-bold text-sm uppercase text-muted-foreground">
                Message {i + 1}: {msg.role}
              </div>

              {/* Thinking Block Editor (if present) */}
              {msg.role === "assistant" && thinking ? (
                <ThinkingEditor 
                  thinking={thinking} 
                  onChange={(newThinking) => handleThinkingChange(i, newThinking)}
                  // onRegenerate will be hooked up later
                />
              ) : (
                <div className="space-y-2">
                  <Label>Content</Label>
                  <Textarea 
                    value={msg.content || ""} 
                    onChange={(e) => updateMessage(i, { ...msg, content: e.target.value })}
                    className="min-h-[150px]"
                  />
                </div>
              )}

              {/* Tool Calls */}
              {msg.tool_calls && msg.tool_calls.length > 0 && (
                <div className="space-y-4 mt-4 pl-4 border-l-2">
                  <h3 className="font-semibold text-sm">Tool Calls</h3>
                  {msg.tool_calls.map((toolCall, j) => (
                    <ToolEditor 
                      key={j}
                      toolCall={toolCall}
                      onChange={(newToolCall) => {
                        const newToolCalls = [...(msg.tool_calls || [])];
                        newToolCalls[j] = newToolCall;
                        updateMessage(i, { ...msg, tool_calls: newToolCalls });
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

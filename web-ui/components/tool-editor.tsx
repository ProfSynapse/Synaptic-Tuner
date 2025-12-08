"use client"

import { useState, useEffect } from "react";
import { ToolCall } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ToolEditorProps {
  toolCall: ToolCall;
  onChange: (newToolCall: ToolCall) => void;
}

export function ToolEditor({ toolCall, onChange }: ToolEditorProps) {
  const [argsString, setArgsString] = useState(toolCall.function.arguments);
  const [isValidJson, setIsValidJson] = useState(true);

  useEffect(() => {
    setArgsString(toolCall.function.arguments);
  }, [toolCall.function.arguments]);

  const handleArgsChange = (value: string) => {
    setArgsString(value);
    try {
      JSON.parse(value);
      setIsValidJson(true);
      onChange({
        ...toolCall,
        function: {
          ...toolCall.function,
          arguments: value
        }
      });
    } catch (e) {
      setIsValidJson(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">Tool Call</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>ID</Label>
            <Input 
              value={toolCall.id} 
              onChange={(e) => onChange({ ...toolCall, id: e.target.value })} 
            />
          </div>
          <div className="space-y-2">
            <Label>Function Name</Label>
            <Input 
              value={toolCall.function.name} 
              onChange={(e) => onChange({ 
                ...toolCall, 
                function: { ...toolCall.function, name: e.target.value } 
              })} 
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label className={!isValidJson ? "text-red-500" : ""}>
            Arguments (JSON) {!isValidJson && "- Invalid JSON"}
          </Label>
          <Textarea 
            value={argsString} 
            onChange={(e) => handleArgsChange(e.target.value)}
            className={`font-mono text-sm h-32 ${!isValidJson ? "border-red-500 focus-visible:ring-red-500" : ""}`}
          />
        </div>
      </CardContent>
    </Card>
  );
}

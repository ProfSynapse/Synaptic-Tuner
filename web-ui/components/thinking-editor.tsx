"use client"

import { ThinkingBlock } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Wand2 } from "lucide-react";

interface ThinkingEditorProps {
  thinking: ThinkingBlock;
  onChange: (newThinking: ThinkingBlock) => void;
  onRegenerate?: () => void;
  isRegenerating?: boolean;
}

export function ThinkingEditor({ thinking, onChange, onRegenerate, isRegenerating }: ThinkingEditorProps) {
  
  const handleChange = (field: keyof ThinkingBlock, value: any) => {
    onChange({ ...thinking, [field]: value });
  };

  const handleArrayChange = (field: 'requirements' | 'plan', value: string) => {
    const lines = value.split('\n').filter(line => line.trim());
    onChange({ ...thinking, [field]: lines });
  };

  return (
    <Card className="border-blue-200 dark:border-blue-900">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-blue-600 dark:text-blue-400">
          Thinking Block
        </CardTitle>
        {onRegenerate && (
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onRegenerate}
            disabled={isRegenerating}
          >
            <Wand2 className="mr-2 h-4 w-4" />
            {isRegenerating ? "Regenerating..." : "Regenerate"}
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Goal</Label>
          <Input 
            value={thinking.goal} 
            onChange={(e) => handleChange('goal', e.target.value)} 
          />
        </div>

        <div className="space-y-2">
          <Label>Memory</Label>
          <Textarea 
            value={thinking.memory} 
            onChange={(e) => handleChange('memory', e.target.value)}
            className="h-24"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Requirements (one per line)</Label>
            <Textarea 
              value={thinking.requirements.join('\n')} 
              onChange={(e) => handleArrayChange('requirements', e.target.value)}
              className="h-32"
            />
          </div>
          <div className="space-y-2">
            <Label>Plan (one per line)</Label>
            <Textarea 
              value={thinking.plan.join('\n')} 
              onChange={(e) => handleArrayChange('plan', e.target.value)}
              className="h-32"
            />
          </div>
        </div>

        <div className="flex items-center gap-8">
          <div className="space-y-2">
            <Label>Confidence</Label>
            <Input 
              type="number" 
              step="0.01" 
              min="0" 
              max="1"
              value={thinking.confidence}
              onChange={(e) => handleChange('confidence', parseFloat(e.target.value))}
              className="w-32"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Checkbox 
              id="complex" 
              checked={thinking.assessment.complex}
              onCheckedChange={(checked) => 
                handleChange('assessment', { ...thinking.assessment, complex: checked })
              }
            />
            <Label htmlFor="complex">Complex</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox 
              id="risky" 
              checked={thinking.assessment.risky}
              onCheckedChange={(checked) => 
                handleChange('assessment', { ...thinking.assessment, risky: checked })
              }
            />
            <Label htmlFor="risky">Risky</Label>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

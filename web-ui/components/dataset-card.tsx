import Link from "next/link";
import { FileText, Database } from "lucide-react";
import { DatasetFile } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface DatasetCardProps {
  dataset: DatasetFile;
}

export function DatasetCard({ dataset }: DatasetCardProps) {
  // Encode the path to be URL-safe
  const encodedPath = encodeURIComponent(dataset.path);

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <div className="flex justify-between items-start">
          <CardTitle className="text-lg truncate" title={dataset.name}>
            {dataset.name}
          </CardTitle>
          <Badge variant={dataset.category === "Behaviors" ? "default" : "secondary"}>
            {dataset.category}
          </Badge>
        </div>
        <CardDescription className="truncate" title={dataset.path}>
          {dataset.path.split(/[\\/]/).pop()}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-muted-foreground" />
            <span>{dataset.count} examples</span>
          </div>
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-muted-foreground" />
            <span>{dataset.size} KB</span>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Link href={`/datasets/${encodedPath}`} className="w-full">
          <Button className="w-full">Open Editor</Button>
        </Link>
      </CardFooter>
    </Card>
  );
}

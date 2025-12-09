import { getDatasets } from "@/lib/actions/datasets";
import { DatasetCard } from "@/components/dataset-card";

export default async function DatasetsPage() {
  const datasets = await getDatasets();

  // Group by category
  const behaviors = datasets.filter((d) => d.category === "Behaviors");
  const toolsets = datasets.filter((d) => d.category === "Toolsets");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dataset Library</h1>
        <p className="text-muted-foreground">
          Manage and edit your training datasets.
        </p>
      </div>

      {behaviors.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Behavior Datasets</h2>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {behaviors.map((dataset) => (
              <DatasetCard key={dataset.path} dataset={dataset} />
            ))}
          </div>
        </div>
      )}

      {toolsets.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Toolset Datasets</h2>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {toolsets.map((dataset) => (
              <DatasetCard key={dataset.path} dataset={dataset} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

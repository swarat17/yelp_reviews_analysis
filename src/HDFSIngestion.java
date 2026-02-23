import java.io.*;

public class HDFSIngestion {
    public static void main(String[] args) {
        // Define dataset paths inside the container
        String localDatasetPath = "/data/yelp_academic_dataset_review.json";  // Path inside the container
        String hdfsTargetPath = "/input"; // HDFS directory

        try {
            // Step 1: Create HDFS directory (if not exists)
            System.out.println("Creating HDFS directory: " + hdfsTargetPath);
            executeCommand("hdfs dfs -mkdir -p " + hdfsTargetPath);

            // Step 2: Upload file to HDFS
            System.out.println("Uploading dataset to HDFS...");
            executeCommand("hdfs dfs -put " + localDatasetPath + " " + hdfsTargetPath);

            // Step 3: Verify ingestion
            System.out.println("Listing files in HDFS directory:");
            executeCommand("hdfs dfs -ls " + hdfsTargetPath);

            System.out.println("Data Ingestion Completed Successfully!");

        } catch (Exception e) {
            System.err.println("Error during HDFS ingestion: " + e.getMessage());
        }
    }

    // Function to execute shell commands inside the container
    private static void executeCommand(String command) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("bash", "-c", command);
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Command failed with exit code: " + exitCode);
        }
    }
}
import OpenAI from "openai";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import csvParser from "csv-parser";
import { Readable } from "stream";

dotenv.config({ path: path.resolve(__dirname, "../config/config.env") });

const csvFile = path.resolve(__dirname, "../dataset/NSE_BANKING_SECTOR.csv");
const jsonlFile = path.resolve(
  __dirname,
  "../dataset/train_prompt_completion_pairs.jsonl"
);
const trainjsonlFile = path.resolve(
  __dirname,
  "../dataset/train_prompt_completion_pairs.jsonl"
);
const valjsonFile = path.resolve(
  __dirname,
  "../dataset/val_prompt_completion_pairs.jsonl"
);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const chatWithRegularModel = async (
  userRequest: string,
  systemContent: string = "You are a helpful assistant"
) => {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "user", content: systemContent },
        { role: "assistant", content: userRequest },
      ],
    });

    const chatResponse = response.choices[0].message.content;
    return chatResponse;
  } catch (error) {
    console.error(`Error with open ai`, error);
  }
};

const replaceNewLinesWithSpaces = (text: string): string => {
  return text.replace(/\n/g, "").trim();
};

const generatePromptCompletionPairs = async (
  inputCsvFile: string,
  outputJsonlFile: string
): Promise<void> => {
  const promptCompletionPairs: any[] = [];

  // Read the CSV file
  const rows: any[] = await new Promise((resolve, reject) => {
    const results: any[] = [];
    fs.createReadStream(inputCsvFile)
      .pipe(csvParser())
      .on("data", (row) => results.push(row))
      .on("end", () => resolve(results))
      .on("error", (err) => reject(err));
  });

  // Process each row
  for (const row of rows) {
    // Generate first prompt-completion pair
    let prompt = `
            Please provide a summary of the stock price for
            ${row["SYMBOL"]} on ${row["DATE"]}.
        `;

    let completion = `
            On ${row["DATE"]}, the stock price variations (${row["SYMBOL"]}) was as follows: 
            Previous Close: ${row["PREV CLOSE"]} 
            Open: ${row["OPEN"]} 
            High: ${row["HIGH"]} 
            Low: ${row["LOW"]} 
            Last: ${row["LAST"]} 
            Close: ${row["CLOSE"]} 
            VWAP: ${row["VWAP"]} 
            Volume: ${row["VOLUME"]} 
            Turnover: ${row["TURNOVER"]} 
            Trades: ${row["TRADES"]} 
            Deliverable Volume: ${row["DELIVERABLE VOLUME"]} 
            Percentage Deliverable: ${row["%DELIVERBLE"]}
        `;

    prompt = replaceNewLinesWithSpaces(prompt);
    completion = replaceNewLinesWithSpaces(completion);

    promptCompletionPairs.push({
      messages: [
        { role: "user", content: prompt },
        { role: "assistant", content: completion },
      ],
    });

    // Generate second prompt-completion pair
    prompt = `What was the high and low price for ${row["SYMBOL"]} on ${row["DATE"]}.`;

    completion = `
            On ${row["DATE"]}, the high and low price for (${row["SYMBOL"]}) was as follows: 
            High: ${row["HIGH"]} 
            Low: ${row["LOW"]} 
        `;

    prompt = replaceNewLinesWithSpaces(prompt);
    completion = replaceNewLinesWithSpaces(completion);

    promptCompletionPairs.push({
      messages: [
        { role: "user", content: prompt },
        { role: "assistant", content: completion },
      ],
    });
  }

  // Write to JSONL file
  const jsonlStream = fs.createWriteStream(outputJsonlFile);
  for (const pair of promptCompletionPairs) {
    jsonlStream.write(JSON.stringify(pair) + "\n");
  }
  jsonlStream.end();

  console.log(`
        Generated ${promptCompletionPairs.length} prompt-completion pairs and
        exported to ${outputJsonlFile}
    `);
};

const splitData = (
  inputJsonlFile: string,
  trainOutputFile: string,
  validOutputFile: string,
  splitRatio: number = 0.8
): void => {
  // Read data from the input JSONL file
  const data = fs
    .readFileSync(inputJsonlFile, "utf-8")
    .split("\n")
    .filter((line) => line.trim() !== "");

  // Shuffle the data
  const shuffledData = data.sort(() => Math.random() - 0.5);

  // Calculate split index
  const splitIndex = Math.floor(shuffledData.length * splitRatio);

  // Split into train and validation sets
  const trainData = shuffledData.slice(0, splitIndex);
  const validData = shuffledData.slice(splitIndex);

  // Write train data to output file
  fs.writeFileSync(trainOutputFile, trainData.join("\n"));

  // Write validation data to output file
  fs.writeFileSync(validOutputFile, validData.join("\n"));

  // Log the results
  console.log(`
      Split data into training (${trainData.length} samples) and
      validation (${validData.length} samples).
    `);
  console.log(`Training data exported to ${trainOutputFile}`);
  console.log(`Validation data exported to ${validOutputFile}`);
};

//trying to test the model generically
(async () => {
  const userRequest =
    "please provide a summary of the stock data for the HDFC Bank on January 1, 2016";
  const response = await chatWithRegularModel(userRequest);
  console.log(response);
})();


//split the data into training and testing data
(() => {
    generatePromptCompletionPairs(csvFile, jsonlFile)
    splitData(jsonlFile, trainjsonlFile, valjsonFile)
})();


//function that uploads the training data
(async () => {
    const response = await openai.files.create({
        file: fs.createReadStream(trainjsonlFile),
        purpose: 'fine-tune'
    })

    console.log(response)
})();


//function that uploads the validation data
(async () => {
    const response = await openai.files.create({
        file: fs.createReadStream(valjsonFile),
        purpose: 'fine-tune'
    })

    console.log(response)
})();


//function that starts the fine tuning
//remember this is just an example of how it would work, it's not how you'd do it in a real example
(async () => {
  console.log("beginning fine tuning.....");
  try {
    const response = await openai.fineTuning.jobs.create({
      model: 'gpt-3.5-turbo',
      training_file: process.env.TRAIN_FILE_ID as string,
      validation_file: process.env.TEST_FILE_ID as string,
      suffix: "bank_stocks",
    });

    console.log(response);
  } catch (error) {
    console.error("Error during fine tuning ", error);
  }
})();


//funvtion to visualise the training and validation loss from the metrics file
const retrieveMetrics = async (fileId: string): Promise<any[]> => {
  const response = await openai.files.content(fileId);

  const readableStream = response.body as unknown as Readable;

  return new Promise((resolve, reject) => {
    const results: any[] = [];

    // Use csv-parser to parse the content
    readableStream
      .pipe(csvParser())
      .on("data", (row) => results.push(row))
      .on("end", () => resolve(results))
      .on("error", (error) => reject(error));
  });
};

const chatWithFineTunedModel = async (userRequest: string) => {
  const fineTuneJob = await openai.fineTuning.jobs.retrieve(
    process.env.FINE_TUNE_JOB_ID as string
  );

  try {
    if (!fineTuneJob.fine_tuned_model) {
      throw new Error(
        "Fine-tuned model not found. Ensure the fine-tuning job is complete."
      );
    }

    const response = await openai.chat.completions.create({
      model: fineTuneJob.fine_tuned_model,
      messages: [{ role: "user", content: userRequest }],
    });

    const chatResponse = response.choices[0].message.content;
    return chatResponse;
  } catch (error) {
    console.error(`Error with open ai`, error);
  }
};

(async () => {
  const userRequest =
    "please provide a summary of the stock data for the HDFC Bank on January 1, 2016";
  const response = await chatWithFineTunedModel(userRequest);
  console.log(response);
})();

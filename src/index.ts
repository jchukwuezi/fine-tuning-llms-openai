import express, {Application } from "express";
import cors from 'cors'
import dotenv from 'dotenv'
import userRoutes from './routes/api/user.routes'
import { connectDB } from "./config/db";

dotenv.config({path: './config/config.env'})

//this project has no database
//mongoose not being used
//connectDB()

const app: Application = express()
const PORT = process.env.PORT || 8080;

app.use(cors({
    origin: ['', ''],
    credentials: true
}))

app.use(express.json())

app.use('/api/users', userRoutes)

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`)
})





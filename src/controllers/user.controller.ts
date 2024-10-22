import { Request, Response } from "express";
import { User } from "../models/user.model";
import bcrypt from "bcrypt";

const registerUser = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    const { name, email, password } = req.body;

    //validate required fields
    if (!name || !email || !password) {
      res
        .status(400)
        .json({ message: "Name, email, and password are required." });
      return;
    }

    // Check if a user with the given email already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      res.status(409).json({ message: "User with this email already exists." });
      return;
    }

    //hash password
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    const newUser = new User({
      name,
      email,
      password: hashedPassword,
    });

    await newUser.save();
  } catch (error) {
    res.status(500).json({ message: "Error registering user", error });
  }
};

const loginUser = async (
  req: Request,
  res: Response
): Promise<void> => {};

export const userController = {
    registerUser,
    loginUser
}
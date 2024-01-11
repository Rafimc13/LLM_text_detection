import pandas as pd
import re
import uuid
import random
from GPT_Prompting import PromptingGPT
from tqdm import tqdm



class Generate_Essay:

    def generate_ids(self, train_data_df):
        """
        Creating unique id's for the new generated essays
        from the LLM. Use of library uuid
        :param: train_data_df: dataframe,
        :return: new_id: str (6 chars)
        """
        df_columns = train_data_df.columns.tolist()
        existing_ids = train_data_df[df_columns[0]].tolist()
        new_id = str(uuid.uuid4())[:8]
        while new_id in existing_ids:
            new_id = str(uuid.uuid4())[:8]

        return new_id

    def create_new_row(self, train_data_df, id, prompt_id, essay, generated=1):
        """
        Create a new row iot to insert it in the train data daraframe.
        :param train_data_df: dataframe,
        :param id: str,
        :param prompt_id: int,
        :param essay: str,
        :param generated: int,
        :return: dictionary
        """
        train_columns = train_data_df.columns.tolist()
        new_row_dict = {
            train_columns[0]: id,
            train_columns[1]: prompt_id,
            train_columns[2]: essay,
            train_columns[3]: generated,
        }

        return new_row_dict

    def choose_random_prompt(self, train_prompt_df, prompt_id):
        """
        Choose randomly one prompt from the many prompts that
        generated via the initial two prompts. Each initial prompt
        has the same number of generated in order the final number
        of generated essays to be balanced.
        :param train_prompt_df: dataframe
        :param prompt_id: int
        :return: string
        """
        prompt_cols = prompts_df.columns.tolist()
        selected_prompts = prompts_df[prompts_df[prompt_cols[0]] == prompt_id][prompt_cols[2]]
        random_prompt = selected_prompts.sample().iloc[0]

        return random_prompt

    def generate_essays(self, prompts_df, train_data_df):
        """
        Using the GPT API in order to generate new essays based on
        the prompts of file 'train_prompts.csv'. Will be generated the
        same number of essays as the number of student's essays in
        order to balance the train data.
        :param prompts_df: dataframe,
        :param train_data_df: dataframe,
        :return: dataframe
        """
        prompt_cols = prompts_df.columns.tolist()
        train_columns = train_data_df.columns.tolist()
        sum_of_augmented = 0
        for i in range(len(train_data_df)):
            if train_data_df.loc[i, train_columns[3]] == 1:
                sum_of_augmented += 1
        num_of_gen_essays = int((len(train_data_df) - sum_of_augmented)/2)
        with tqdm(total=2) as pbar:  # Check our time and iters remaining!
            for _ in range(2):
                # Create in each iter (inside loop) a new instance because GPT 3.5 has 16,385 limit tokens
                GPT_prompts = PromptingGPT()
                prompt1 = self.choose_random_prompt(train_data_df, 0)
                prompt2 = self.choose_random_prompt(train_data_df, 1)
                choice = random.choices([True, False])  # Give a choice to insert or not the source texts
                # if choice:
                #     prompt1 = prompt1 + ('\n') + prompts_df.loc[0, prompt_cols[3]]
                #     prompt2 = prompt2 + ('\n') + prompts_df.loc[1, prompt_cols[3]]
                # Prompt No1 for prompt_id=0
                prompt1 = prompts_df.loc[0, prompt_cols[2]]
                prompt2 = prompts_df.loc[1, prompt_cols[2]]
                essay1 = GPT_prompts.make_prompts_turbo(prompt1)
                cleaned_essay1 = re.sub(r'\[.*?\]', '', essay1)
                new_id1 = self.generate_ids(train_data_df)
                new_row1 = self.create_new_row(train_data_df, id=new_id1, prompt_id=0, essay=cleaned_essay1)
                train_data_df = train_data_df._append(new_row1, ignore_index=True)
                # Prompt No2 for prompt_id=1
                essay2 = GPT_prompts.make_prompts_turbo(prompt2)
                cleaned_essay2 = re.sub(r'\[.*?\]', '', essay2)
                new_id2 = self.generate_ids(train_data_df)
                new_row2 = self.create_new_row(train_data_df, id=new_id2, prompt_id=1, essay=cleaned_essay2)
                train_data_df = train_data_df._append(new_row2, ignore_index=True)
                pbar.update(1)  # Update the progress bar

        return train_data_df


if __name__ == "__main__":

    # Create an instance of class Text_Clasification
    gendata = Generate_Essay()

    # Read the .csv file of prompts iot take the prompts
    prompts_df = pd.read_csv('data/train_prompts_test.csv')
    prompt_cols = prompts_df.columns.tolist()
    # Read the .csv file of train data
    train_data_df = pd.read_csv('data/train_essays.csv')
    train_data_cols = train_data_df.columns.tolist()

    # Creation of new essays based on augmentation of LLM
    complete_train_data_df = gendata.generate_essays(prompts_df, train_data_df)
    complete_train_data_df = complete_train_data_df.set_index(train_data_cols[0])

    # Save the complete training set of essays (students/LLM)
    complete_train_data_df.to_csv('data/all_train_essays_test.csv')
    complete_train_data_df.to_html('data/all_train_essays_test.html')

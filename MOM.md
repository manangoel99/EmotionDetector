# Minutes of Meetings

## 19th March, 2020

- Discuss ideas for project
- Narrowed down three ideas
- Talked about requirements for each idea
- Takeaway : explore available projects similar to pur ideas

## 31st March, 2020

- Share progress so far
  - Screenshare and show the basic app + upload mechanism
  - Screenshare and show the different kinds of face datasets
- Decide baseline model for face detection
- Decide main functionalities of dashboard
- Divide work
- Takeaway : Implement baseline model, draft dashboard and authentication 

## 5th April, 2020

- Share progress so far
  - Screenshare and show dashboard
  - Share model accuracy
- Discuss database schema
- Discuss ways of making project easily deployable (Dockerise)
- Discuss ways to make the system distributed (task queueing, parallel processing)
- Takeaway : Dockerise project, link psql database, explore use of Ray/Celery in distributed computing

## 12th April, 2020

- Share progress so far
  - Screenshare and show the database schema and few entries
  - Share time improvements from baseline model to distributed model
  - Get everyone to install project using Docker
- Discuss how we want to show the output (edited video + graphs - pie chart and line graph)
- Discuss how to integrate front end and backend - where to store uploaded and processed videos, how to show status
- Decide UI - which all pages should exist, what should each page show
- Can we use either only Celery or only Ray to do both task queueing and paralleling frame processing?
- Takeaway : Explore use of ray, develop frontend

## 16th April, 2020

- Share progress so far
  - Screenshare and show the UI + the user experience for uploading & watching the progress of the video
  - Screenshare and understand how the routing and db calls are integrated
  - Celebrate that we have an end-to-end working model!!
- Discuss User experience changes - add error comments for file extensions, show dynamic progress of processing videos etc
- Split unit testing and decide who will cover which module
- Discuss ways of making project more modular
- Discuss more areas where we need to handle errors
- Takeaway : Write unit tests, split project into mor modules, handle discussed errors

## 20th April, 2020

- Full tour of the app
- Quick review of code
   - error handling (one new case)
   - try/catch statements added (all handled)
   - see if we're missing out some major tests (all handled)
- Discuss on how to make project cleaner - make separate folder for docs, css files and images
- Test app on mobile just to make sure functionalities are working
- Takeaway : test project individually one last time

## 24th April, 2020

- Add comments and clean code
- Prep for presentation after this.

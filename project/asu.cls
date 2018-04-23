\newcommand{\asuthesisupdate}{2016/07/31}
\newcommand{\asuthesisversion}{v1.2}
\newcommand{\asuthesisauthor}{Jay Fenwick}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{asu}

\newcommand{\@thesisptsz}{}
\newcommand{\@thesisside}{}
\newcommand{\@thesisopen}{}
\newcommand{\@thesislnsp}{}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[\@thesisptsz,\@thesisside,\@thesisopen]{report}

\newcommand{\bachelordegree}[1]{\ifx\empty#1\else\gdef\@bachelordegree{#1}\fi}
\newcommand{\degree}[1]{\ifx\empty#1\else\gdef\@degree{#1}\fi}
\newcommand{\degreeabbrev}[1]{\ifx\empty#1\else\gdef\@degreeabbrev{#1}\fi}
\newcommand{\department}[1]{\ifx\empty#1\else\gdef\@department{#1}\fi}
\newcommand{\gradmonth}[1]{\ifx\empty#1\else\gdef\@gradmonth{#1}\fi}
\newcommand{\gradyear}[1]{\ifx\empty#1\else\gdef\@gradyear{#1}\fi}
\newcommand{\thesischair}[1]{\ifx\empty#1\else\gdef\@thesischair{#1}\fi}
\newcommand{\thesismemberone}[1]{\ifx\empty#1\else\gdef\@thesismemberone{#1}\fi}
\newcommand{\thesismembertwo}[1]{\ifx\empty#1\else\gdef\@thesismembertwo{#1}\fi}
\newcommand{\thesismemberthree}[1]{\ifx\empty#1\else\gdef\@thesismemberthree{#1}\fi}
\newcommand{\deptchair}[1]{\ifx\empty#1\else\gdef\@deptchair{#1}\fi}
\newcommand{\dean}[1]{\ifx\empty#1\else\gdef\@dean{#1}\fi}
\newcommand{\authorcaps}[1]{\ifx\empty#1\else\gdef\@authorcaps{#1}\fi}
\renewcommand{\author}[1]{\ifx\empty#1\else\gdef\@author{#1}\fi}
\renewcommand{\title}[1]{\ifx\empty#1\else\gdef\@title{#1}\fi}

\newcommand{\newlinestretch}[1]{\renewcommand{\baselinestretch}{#1}\normalsize}

\newenvironment{preliminary}
  {\pagestyle{plain}\pagenumbering{roman}}
  {\cleardoublepage\pagenumbering{arabic}}

\renewcommand{\maketitle}{
   \begin{titlepage}
   \vspace*{0.5in}
   \begin{center}
      {\@title} \\[25ex]
      \newlinestretch{1}
      A Project \\
      \newlinestretch{1}
      by \\
      {\@authorcaps} \\[6ex]
      Submitted to the Graduate School \\
      \newlinestretch{1}
      Appalachian State University \\
      in partial fulfillment of the requirements for the degree of \\
      {\@degree} \\[12ex]
      {\@gradmonth} {\@gradyear} \\
      Department of {\@department}
   \end{center}
   \setcounter{page}{1}
   \end{titlepage}
   \setcounter{page}{2}
}

\newcommand{\makeapproval}{
   \thispagestyle{empty}
   \vspace*{0.5in}
   \begin{center}
      {\@title} \\
      \vspace{1in}
      \newlinestretch{1}
      A Thesis \\
      \newlinestretch{1}
      by \\
      {\@authorcaps} \\
      {\@gradmonth} {\@gradyear}
   \end{center}
   \vspace{3cm}
   APPROVED BY: \\ [4ex]
   \underline{\hspace{2.7in}} \\
   {\@thesischair} \\
   Chairperson, Thesis Committee \\[4ex]
   \underline{\hspace{2.7in}} \\
   {\@thesismembertwo} \\
   Member, Thesis Committee \\[4ex]
   \underline{\hspace{2.7in}} \\
   {\@thesismemberthree} \\
   Member, Thesis Committee \\[4ex]
   \underline{\hspace{2.7in}} \\
   {\@deptchair} \\
   Chairperson, Department of {\@department} \\[4ex]
   \underline{\hspace{2.7in}} \\
   {\@dean} \\
   Dean, Cratis D. Williams School of Graduate Studies
   \newpage
}

\newcommand{\makecopyright}{
    \vspace*{3.5in}
    \begin{center}
    Copyright{\copyright} {\@author} {\@gradyear}\\
    All Rights Reserved
    \end{center}
    \setcounter{page}{3}
    \newpage
}

\renewenvironment{abstract}{
   \vspace*{0.5in}
   \begin{center}
      {\bf Abstract} \\[2ex]
      \newlinestretch{2}
       {\@title}\\
                     
      {\@author}\\
      \newlinestretch{1}
      {\@bachelordegree}\\
      {\@degreeabbrev}, Appalachian State University\\
      \newlinestretch{2}
      Chairperson:  {\@thesischair}
   \end{center}
   \newlinestretch{2}
}{\par\newpage}

\newenvironment{acknowledgement}{
   \vspace*{0.5in}
   \begin{center}
      ACKNOWLEDGEMENTS \\[2ex]
      \newlinestretch{2}
   \end{center}
   \newlinestretch{2}
}{\par\newpage}

\newenvironment{dedication}{
   \vspace*{0.5in}
   \newlinestretch{2}
}{\par\newpage}

%
%  PAGE STYLE
%
\newcommand{\ps@asu}{%
   \renewcommand{\@oddhead}{%
      \begin{minipage}{\textwidth}\normalfont
          \hfill \thepage
      \end{minipage}}
   \renewcommand{\@evenhead}{\@oddhead}
   \renewcommand{\@oddfoot}{}
   \renewcommand{\@evenfoot}{}
}
\pagestyle{asu}
\setlength{\parindent}{0.5in}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}


%
%  MARGINS
%
%  Sets margins to 1 1/2" left margin and 1" everywhere else.
%
\setlength{\voffset}{-1in}
\setlength{\hoffset}{-1in}

\setlength{\textheight}{\paperheight}
\setlength{\textwidth}{\paperwidth}

\addtolength{\textheight}{-2in}
\setlength{\topmargin}{1in}
\setlength{\headsep}{.5in}
\setlength{\footskip}{.5in}

\addtolength{\textwidth}{-2.5in}
\setlength{\oddsidemargin}{1.5in}
\setlength{\evensidemargin}{1in}
\setlength{\marginparwidth}{.75in}
\setlength{\marginparsep}{.125in}

\flushbottom


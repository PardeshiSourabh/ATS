from flask import Flask, render_template, url_for, flash, redirect, request
from form import Registration, LoginForm
from calculate import Calculate

app = Flask(__name__)
app.config['SECRET_KEY'] = '0332dde67326bc2e'
app.config['JSON_AS_ASCII'] = False


@app.route("/")
@app.route("/job_home")
def home():
    return render_template('job_home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = Registration()
    if form.validate_on_submit():
        flash("User account for {} created successfully!".format(form.username.data), 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash("Login successfull!", 'success')
        return redirect(url_for('home'))
    return render_template('login.html', title='LogIn', form=form)


@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    if request.method == "POST":
        return render_template('analyze.html')


@app.route("/result", methods=['GET', 'POST'])
def result():
    resume = request.form['resume']
    jobdesc = request.form['jobdesc']
    cal = Calculate(resume, jobdesc)

    return render_template('result.html', title="Results", cal=cal)


if __name__ == '__main__':
    app.run(debug=True)

select male.average_age_male, average_age_female
from

(select avg (age) as average_age_male
from customer c 
where gender = 1) as male,

(select avg (age) as average_age_female
from customer c 
where gender = 0) as female;




